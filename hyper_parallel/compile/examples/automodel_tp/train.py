#!/usr/bin/env python3
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
    PassConfig,
    PassPlan,
)
from hyper_parallel.distributed.mesh import MeshContext  # pylint: disable=C0413
from hyper_parallel.distributed import (  # pylint: disable=C0413
    ShardingPlanner,
    apply_sharding_plan,
)

from model import (  # pylint: disable=C0413
    build_model as build_model_from_config,
    DataSampler,
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
            if dist.get_rank() == 0:
                print(f"[setup] backend=hccl, world_size={dist.get_world_size()}, "
                      f"npu_device={torch.npu.current_device()}")
            return "npu"
        dist.init_process_group(backend="gloo")
        if dist.get_rank() == 0:
            print(f"[setup] backend=gloo, world_size={dist.get_world_size()}")
        return "cpu"
    return "cpu"


def build_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    """Build the model on ``device``.

    Delegates to ``build_model`` from ``model.py``, which creates an
    ``AutoModelAdapterForCausalLM`` — a structurally correct model where
    ``rotary_emb`` lives inside ``Attention.forward`` (not in the inner
    model's forward). This means cos/sin are computed *after* the SP
    boundary all-gather, so they naturally have the full sequence length
    — no runtime patch needed.
    """
    return build_model_from_config(cfg["model"], device)


def build_mesh_context(
    parallel_cfg: dict, device_type: str
) -> MeshContext:
    """Build the automodel (dp, cp, tp) mesh + the dense FSDP sub-mesh."""
    tp = parallel_cfg["tp_size"]
    dp = parallel_cfg["dp_size"]
    sequence_parallel = parallel_cfg.get("sequence_parallel", False)
    loss_parallel = parallel_cfg.get("loss_parallel", False)
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
        sequence_parallel=sequence_parallel,
        loss_parallel=loss_parallel,
    )
    ctx.build_meshs(device_type, world_size)
    ctx.tp_rank = dist.get_rank() % tp
    ctx.dp_rank = dist.get_rank() // tp
    return ctx


def train_fn(model, input_ids, labels):
    """Standard CE loss on the boundary-wrapped model forward.

    In SP mode the lm_head boundary all-gathers hidden_states to full
    sequence, so logits and labels both have the full seq_len.  The
    min_seq alignment is a safety no-op.
    """
    outputs = model(input_ids)
    if hasattr(outputs, "logits"):
        logits = outputs.logits
    elif isinstance(outputs, dict) and "logits" in outputs:
        logits = outputs["logits"]
    else:
        logits = outputs[0]
    # Align sequence dimension: logits may be shorter than labels in SP mode
    min_seq = min(logits.shape[1], labels.shape[1])
    logits = logits[:, :min_seq, :]
    labels = labels[:, :min_seq]
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
    tp_size = cfg["parallel"]["tp_size"]
    sequence_parallel = cfg["parallel"].get("sequence_parallel", False)
    loss_parallel = cfg["parallel"].get("loss_parallel", False)
    model = build_model(cfg, device)

    # 2. automodel mesh: (dp, cp, tp) + dense FSDP sub-mesh. TP group ready.
    mesh_ctx = build_mesh_context(cfg["parallel"], device_type)

    # 3. TP sharding plan + production apply. Parameters become plain TP
    #    shards; module forwards are wrapped with boundary TP collectives on
    #    activations. When sequence_parallel=True the boundary I/O contract
    #    changes: attention/mlp boundaries use Shard(1)->Replicate on entry
    #    and Partial->Shard(1) (reduce_scatter) on exit.
    planner = ShardingPlanner()
    plan = planner.plan(
        model,
        mesh_ctx.device_mesh,
        tp_size=cfg["parallel"]["tp_size"],
        sequence_parallel=sequence_parallel,
        loss_parallel=loss_parallel,
    )
    model, _ = apply_sharding_plan(model, plan, mesh_ctx, validate_mode=False)
    if rank == 0:
        print(
            "[automodel] TP sharding applied — params are now plain TP shards, "
            "boundary forwards wrap TP collectives on activations"
        )
        if sequence_parallel:
            print(f"[SP] sequence_parallel=True, tp_size={tp_size}")

    # 3a. No rotary_emb patch needed — the model places
    #     rotary_emb inside Attention.forward (after boundary all-gather),
    #     so cos/sin naturally have the full sequence length.
    if sequence_parallel and rank == 0:
        print("[SP] Using AutoModelAdapterForCausalLM — rotary_emb inside attention, no patch needed")

    # 4. FSDP wrap plan for the graph-mode FSDPPass (wrap everything).
    fsdp_plan = PassPlan()
    fsdp_plan.fsdp_wrap_pattern("*")

    # 5. GraphTrainer reuses the automodel mesh (tp group already created);
    #    it registers the dp sub-mesh as "fsdp" and back-fills fsdp_degree.
    tcfg = cfg["train"]
    trainer = GraphTrainer(
        model=model,
        train_fn=train_fn,
        pass_config=PassConfig(
            enable_overlap=cfg["parallel"].get("enable_overlap", True),
            fsdp_degree=cfg["parallel"]["dp_size"],
            tp_size=tp_size,
            sequence_parallel=sequence_parallel,
            loss_parallel=loss_parallel,
        ),
        pass_plan=fsdp_plan,
        optimizer_config={"lr": tcfg.get("lr", 1e-4),
                          "grad_clip": tcfg.get("grad_clip", 1.0)},
        mesh_context=mesh_ctx,
        device=device,
    )

    # 6. Dummy data + explicit compile so we can inspect the graph.
    #    DataSampler yields full-sequence data; SP sharding is handled by
    #    the embedding boundary's reduce-scatter on hidden_states.
    mcfg = cfg["model"]
    sampler = DataSampler(
        vocab_size=mcfg["vocab_size"],
        batch_size=tcfg["batch_size"],
        seq_len=tcfg["seq_len"],
        max_steps=tcfg["max_steps"],
        tp_size=tp_size,
        tp_rank=mesh_ctx.tp_rank,
        sequence_parallel=sequence_parallel,
        device=device,
    )

    # Sample one batch for compilation (shape must match runtime batches)
    sample_input, sample_label = sampler.sample()
    if sequence_parallel and rank == 0:
        print(f"[SP] sample input full seq_len={sample_input.shape[1]} (SP sharding via embedding reduce-scatter)")
    trainer.compile(sample_input, sample_label)
    inspect_graph(trainer)

    # 7. Training data iterator (DataSampler yields full-sequence batches)
    print("\nStarting training...")
    trainer.train(
        iter(sampler),
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
