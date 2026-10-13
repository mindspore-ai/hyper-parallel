# GraphCompiler activation recomputation

This example compares eager training against `none`, `full`, and `selective`
joint-graph execution. It checks loss, every parameter gradient, SGD updates,
and the final CPU/device RNG states at each step. The default tolerance is
zero. Run commands from the HyperParallel repository root with this checkout
installed using `pip install -e .`.

```bash
python -m hyper_parallel.compile.examples.recompute.train --device cpu
python -m hyper_parallel.compile.examples.recompute.train --device cpu --dropout 0.2
python -m hyper_parallel.compile.examples.recompute.train --device npu --dropout 0.2
python -m hyper_parallel.compile.examples.recompute.train --device cpu --selective-policy save_compute
python -m hyper_parallel.compile.examples.recompute.train --device cpu --exclude layers.0
python -m hyper_parallel.compile.examples.recompute.train --device cpu --dump-dir /tmp/recompute_graphs
```

`--rtol` and `--atol` can set an explicit comparison tolerance for other
hardware. The example is a correctness check, not a throughput benchmark.

## Compiler API

```python
from hyper_parallel.compile import GraphCompiler, PassConfig, RecomputeConfig

config = PassConfig(
    fsdp_enabled=False,
    enable_overlap=False,
    recompute=RecomputeConfig(
        mode="selective",
        module_patterns=("layers.*",),
        selective_policy="eager",
        exclude_module_patterns=("layers.0",),
        save_ops=("aten.mm.default",),
    ),
)
compiler = GraphCompiler(model, train_loss, config, device=device)
loss, loss_dict = compiler.forward_backward(value=value, label=label)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
print(compiler.recompute_report)
```

The training function accepts `(model, **inputs)` and returns a scalar loss or `(loss, loss_dict)`.
Use an unwrapped model: eager checkpoint/swap wrappers are not the graph pass
configuration interface. Input shapes retain GraphCompiler's existing static
shape contract; retrace after changing shapes or recompute configuration.

Configuration details:

- `none` is the default and adds no recompute pass.
- `full` recomputes backward-required pure operations inside selected regions.
  Random, effectful, mutable, opaque and cross-region values are safety exceptions.
- `selective` saves expensive operations and recomputes ordinary operations.
  `eager` alternates mm/matmul/linear/grouped-mm decisions per selected region;
  `save_compute` saves all of these operations. `addmm` and `bmm` are saved in
  both selective modes, matching their treatment in Hyper's eager policy.
  Operator decomposition can change eager-versus-graph policy correspondence.
- Module patterns match full paths in `nn_module_stack`. The outermost matching
  ancestor establishes the region. Excluding a module excludes its subtree.
  Every include pattern must match traced code. Unexecuted branches do not match.
- `save_ops` contains exact traced operator strings. Every entry must occur in
  the selected regions; misspellings produce an error instead of a silent no-op.
- `policy_fn(node)` can override built-in decisions, returning either Hyper's
  or Torch's `CheckpointPolicy`. It receives FX metadata, not real eager tensor
  arguments. SAVE/RECOMPUTE preferences are resolved directly without a solver.
  Safety constraints override preferences, while an unsafe explicit
  `MUST_RECOMPUTE` request fails. `MUST_SWAP` is rejected.

## AutoModel + TextTrainer configuration

The existing `automodel_text_graph/train_lm_graph_tp2_fsdp2.yaml` also accepts:

```yaml
compile:
  enabled: true
  use_joint_graph: true
  recompute:
    mode: selective
    module_patterns: ["model.layers.*"]
    selective_policy: eager
```

`build_pass_config_from_trainer_config` maps this section to `RecomputeConfig`
when `BaseTrainer` constructs `GraphCompiler`. Use the actual decoder FQNs of
pretrained models; `model.layers.*` fits Qwen3's decoder. Keep eager
`activation_checkpoint.mode: off` and `activation_swap: none`.

The launcher forwards dotted CLI overrides, for example:

```bash
bash hyper_parallel/compile/examples/automodel_text_graph/run.sh \
  --model.pretrained_model_name_or_path=/path/to/model \
  --dataset.data_path=/path/to/indexed_text_document \
  '--compile.recompute={mode: full, module_patterns: [model.layers.*]}'
```

The indexed dataset prefix must have corresponding `.bin` and `.idx` files.
The tokenizer metadata in the example must agree with the selected model.
Graph recompute requires both `compile.enabled` and `compile.use_joint_graph`;
eager checkpoint/swap wrappers and pipeline parallelism are rejected.

## How it works

The implementation follows TorchTitan GraphTrainer's tag-then-rematerialize
design, but uses Hyper's `nn_module_stack` and `autograd_backward` metadata.
It does not import TorchTitan or use the private min-cut partitioner APIs.

```text
joint FX trace -> cleanup -> RecomputePass -> FSDPPass -> PpPass -> overlap
```

Recomputation runs before FSDP so its parameter consumers participate in
backward all-gather and storage-free planning. Only dependencies required by
backward consumers are copied, in topological order. Shared dependencies have
one duplicate, and original forward users remain unchanged. Copied nodes retain
module metadata and are marked as backward execution. The pass mutates the same
GraphModule, preserving GraphCompiler's state-input and gradient-output contracts.

In this release, enabling PP and recomputation together is rejected before
tracing. The diagram shows pipeline ordering, not validated PP support.

`compiler.recompute_report` returns a detached copy of the compilation report:

- Number of duplicated and policy-saved nodes.
- Exact nodes saved due to mutation/aliasing, RNG, region boundaries, or other
  safety constraints.
- `retained_activation_bytes_before/after`: a static estimate of unique
  non-input tensor storage crossing forward-to-backward edges. Aliases count
  once; input/parameter storage is excluded. This is **not measured peak device
  memory**: it excludes allocator effects, workspace, temporary backward values,
  and later parallel-pass changes. A policy can yield no savings or cost more
  memory on some models; there is no budget optimizer.

## Eager feature coverage and first-release boundaries

| Eager capability | Graph implementation |
| --- | --- |
| Full and selective checkpointing | Supported for traceable selected regions |
| Alternating matmul saves | Supported per selected region; decomposition can differ |
| Selected layers/submodules | Explicit module patterns |
| Exclude a submodule | Explicit subtree exclusion; no eager replay cache |
| Model adapter safe/no-replay declarations | Translate them to include/exclude patterns manually; no automatic adapter discovery yet |
| RNG replay | Random outputs are saved instead; CPU/NPU dropout parity is tested |
| Inplace/alias protection | Schema-based alias groups and their reads are saved, including backward writes |
| Early stop | Only backward-required dependencies are materialized |
| Custom policy | FX node callback; eager runtime-value-dependent callbacks are not interchangeable |
| CPU swap, input swap, group swap and prefetch | Not implemented; requires explicit graph offload/reload/wait operations |
| Higher-order/custom/communication ops | Saved, not replayed; no recursive rematerialization of nested subgraphs |
| Min-cut or memory-budget optimization | Not implemented |
| Reentrant checkpoint / arbitrary Python contexts / recompute sessions | Not implemented; these are eager execution mechanisms |
| Pipeline parallelism | Explicitly rejected in this release |
| Multi-rank training | AutoModel validation covers TP2 + FSDP2 on four NPUs with FP32 parameters; EP remains unvalidated |

CPU dropout currently traces as an inplace random-mask chain. The pass protects
the entire affected alias group and its readers, not just the random producer.
NPU native dropout is saved as a random op; its deterministic consumers may
still be rematerialized. Both preserve the global RNG progression because no
random operation is rerun.

FSDP composition is tested with simulated two-rank collectives and real storage
frees for both reshard settings. The separate AutoModel fixture below exercises
real TP2 + FSDP2 training with the default reshard setting.
Models with hidden Python side effects, cache mutation, custom effects not
declared by operator schemas, or pre-existing checkpoint wrappers are outside
the supported contract. Select known-safe submodules for such models.

## Validation

```bash
python -m pytest -q tests/ut/compile/test_recompute_pass.py
python -m pytest -q tests/ut/compile
```

Tests cover full/selective numerical parity over three SGD steps, dropout RNG,
subtree exclusion, native and Hyper policies, explicit saves, gradient
accumulation, selector errors, unsupported policies, shared/tuple dependencies,
inplace alias hazards, pass ordering, repeated-pass rejection and FSDP reshard
composition. Example graph dumps allow the `_recomputed` nodes and saved
boundaries to be inspected directly.

## Four-device AutoModel validation

`prepare_automodel` derives four configurations from the existing
`automodel_text_graph/train_lm_graph_tp2_fsdp2.yaml`. It writes a fixed-seed
indexed token fixture, keeps TP2 + FSDP2, sequence length 128, global batch 4,
micro batch 1, and 10 optimizer steps. It sets `model.load_base_model: false`
and `model_init_dtype: float32`, using random initialization of the supplied
model architecture. This tests training correctness, not pretrained loading
or language-model convergence. The decoder selector is `model.layers.*`.

```bash
python -m hyper_parallel.compile.examples.recompute.prepare_automodel \
  --model-assets /path/to/Qwen3-0.6B \
  --output-dir /tmp/recompute_automodel

for policy in none full selective save_compute; do
  export VALIDATION_OUTPUT="/tmp/recompute_automodel/${policy}"
  if [ "$policy" = none ]; then
    unset VALIDATION_REFERENCE
  else
    export VALIDATION_REFERENCE=/tmp/recompute_automodel/none
  fi
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
    torchrun --standalone --nproc_per_node=4 \
    hyper_parallel/compile/examples/recompute/validate_automodel.py \
    "/tmp/recompute_automodel/${policy}.yaml" || exit 1
done
```

`validate_automodel` calls the existing `TextTrainer.train()` and observes its
normal step methods. Each rank records every local parameter and pre-clipping
gradient (shape, dtype, SHA-256 of all bytes), loss, gradient norm, and measured
peak allocated device memory. Against `VALIDATION_REFERENCE`, gradients,
updated parameters and gradient norms must match exactly; loss permits absolute
tolerance `1e-5` (zero relative tolerance). A repeat of the baseline exhibited
loss variations of about `2e-6` while tensor hashes stayed identical. The check
also requires the requested number of steps, finite tensors, and nonzero
rematerialized-node counts for enabled policies.

The JSON evidence is saved under `VALIDATION_OUTPUT/rank<N>.json`. Step 1
includes tracing; inspect steps 2 onward separately for steady-state memory.
CPU copies and hashing add synchronization and overhead, so the observed runs
are not throughput benchmarks. Static retained-activation estimates and actual
peak memory are different measurements; the short-sequence Qwen3 fixture can
show large static savings with little change in measured peak allocation.

Known baseline integration limitations found during validation:

- Qwen3-0.6B pretrained TP loading failed with an `lm_head.weight` local-shape
  mismatch before graph compilation. The fixture therefore uses the supported
  random-init option, retaining the model architecture.
- BF16 parameters with `optimizer.fp32_main_params: true` failed on the first
  optimizer step even with recompute disabled: the main parameter retained the
  pre-graph-FSDP shape while its gradient was already sharded. FP32 parameters
  avoid this optimizer integration issue. Setting only the original example's
  BF16 dtype without main parameters also fails its optimizer precision guard.

These paths require separate upstream integration fixes; the four-device
results here do not establish pretrained-loading or BF16-main-optimizer support.
