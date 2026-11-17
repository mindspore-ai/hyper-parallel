# High-performance modules

`hyper_parallel.components.modules` provides drop-in high-performance modules for Ascend NPU. Modules can be used directly in
Python or declaratively replace existing model modules through Trainer YAML.

## Available modules

| Category | Modules |
| --- | --- |
| Attention | `GroupedQueryAttention`, `GatedGroupedQueryAttention`, `MultiLatentAttention`, `GatedMultiLatentAttention`, `DSAAttention`, `DeepseekV32DSAAttention` |
| MoE and MLP | `GroupedExperts`, `SharedExpert`, `SwiGLUMLP` |
| Normalization | `RMSNorm`, `OffsetRMSNorm` |
| MHC | `MhcPreModule`, `MhcPostModule` |

Reusable high-performance functions are provided separately in [`hyper_parallel.components.functional`](../functional/README.md).

Modules can also be constructed directly:

```python
from hyper_parallel.components.modules import RMSNorm

norm = RMSNorm(hidden_size=4096, eps=1e-6).npu()
output = norm(hidden_states)
```

## Module families

Registered module families select an implementation from explicit configuration rules before model construction.
See [Module Family](README.module_family.md) for GroupedExperts MXFP8 examples, rule semantics, and extension guidance.

## GQA and context parallelism

GQA families own their parameter layouts and select local or CP members from the job topology:

- `GroupedQueryAttention` uses a grouped, fused QKV projection and declares checkpoint conversion.
- `GroupedQueryAttention(fuse_qkv=False)` retains independent Q/K/V projections and their checkpoint names. This preserves
  the early communication windows of the former Qwen3-MoE asynchronous implementation.
- `GatedGroupedQueryAttention` uses the per-query-head sigmoid output gate of Qwen3.5 full attention.
  This does not replace the model's linear-attention layers.

`overlap` changes launch/wait placement within the chosen representation; it does not change projection weights.
The CP members reuse `CPContext`. The original planner/apply boundary performs SP input gathering and
TP output reduction; standalone attention classes contain no TP executor or binding methods.
`ulysses_degree: 1` selects KV AllGather; an omitted degree selects pure Ulysses; an intermediate divisor selects
Ulysses plus KV AllGather. The Ulysses degree must divide both TP-local Q and KV head counts. KV heads are not
replicated to the Q head count. Unsupported head layouts fail during construction.

For example, the existing Qwen3-MoE KV AllGather recipe uses:

```yaml
plan_overrides:
  - match: "model.layers.*.self_attn"
    replace:
      kind: module
      implementation:
        _target_: hyper_parallel.components.modules.GroupedQueryAttention
        ulysses_degree: 1
        overlap: false
```

There is no additional CP forward-wrapper replacement. Set `overlap: true` to enable the supported
communication schedule. Ordinary training YAML does not need a `fuse_qkv` entry; the default representation
remains fused. Choosing independent projections is a separate model-construction decision, not a prerequisite
for CP or overlap.

### Native construction without a source module

Ordinary and gated GQA accept exactly one of `config=` or `module=`. Native construction creates the
projections directly, without instantiating a Transformers attention first:

```python
from types import SimpleNamespace
import torch
from hyper_parallel.components.modules import GroupedQueryAttention

config = SimpleNamespace(
    hidden_size=1024, num_attention_heads=16, num_key_value_heads=4,
    head_dim=64, qk_norm=True, rms_norm_eps=1e-6,
    attention_bias=False, dtype=torch.bfloat16,
)
attention = GroupedQueryAttention(config=config)
# In a distributed job, pass its existing mesh to select and bind the CP member:
# attention = GroupedQueryAttention(config=config, mesh=mesh, overlap=True, ulysses_degree=2)
# The model supplies position_embeddings=(cos, sin) at forward time.
```

`GroupedQueryAttention(fuse_qkv=False)` creates independent Q/K/V Linear children through the same family,
local class and CP class. `fuse_qkv=True` is the default; overlap never changes this parameter representation.
`GatedGroupedQueryAttention` creates grouped Q/K/V/gate rows and requires equal
QK and V head dimensions. The gate is always applied before `o_proj`. This constructs an attention component,
not an entire Qwen decoder or its linear-attention layers.

Native math options are explicit:

| Configuration | Meaning and default |
| --- | --- |
| `hidden_size`, `num_attention_heads`, `num_key_value_heads` | Required positive dimensions; Q heads must divide into KV groups |
| `head_dim`, `v_head_dim` | QK width (defaults to `hidden_size / num_attention_heads`) and V width (defaults to QK width) |
| `attention_bias`, `o_proj_bias` | Q/K/V projection bias, default false; output bias defaults to `attention_bias` |
| `qk_norm`, `rms_norm_eps` | Optional per-head Q/K RMSNorm, default false; epsilon defaults to `1e-6` |
| `scaling`, `attention_dropout`, `is_causal`, `sliding_window` | Attention scale (defaults to QK width ** -0.5), dropout (0), causal (true), window (none) |
| `rotary_interleaved` | Rotary layout, default false; supplied cos/sin determine the rotary width, including partial RoPE |
| `dtype`, `layer_idx` | Optional parameter dtype and local cache layer index; device follows the active `torch.device` context |

Native Q/K Norm stores direct scale `gamma`, initialized to one. Norm presence is **not** inferred from
`model_type` or `rms_norm_eps`. Source conversion retains the original Norm convention; in particular,
Qwen3.5 uses `1 + weight`. Copying those raw Norm weights to a native module requires explicit conversion.
Native `state_dict` already uses grouped/split parameter names, so `make_transforms()` returns no HF conversion.
To import an HF checkpoint, use the source-conversion path (and its declared transforms), rather than loading
raw HF tensors as a native checkpoint. Native `reset_parameters()` follows Linear initialization and unit
Norm scales, including after meta materialization; source models retain their own initialization.
Local-to-CP preparation preserves loaded Parameters, aliases and `requires_grad`; it does not reinitialize them.

Packed attention accepts global, batch-major cumulative lengths through the common FA interface, for example
`actual_seq_len=(100, 300)` for documents of length 100 and 200. CP converts these boundaries into each query
owner's document fragments; the FA call uses TND and preserves document isolation. Ordinary CP causal attention
without an explicit mask also uses document/prefix metadata instead of creating a global square mask.

Do not combine packed lengths with an explicit attention mask. At the full HF model boundary, select a model
input adapter that forwards the lengths and suppresses the model's own dense-mask builder. Attention-level TND
support alone does not change Trainer data preparation. Existing Qwen3-MoE online examples still use their
explicit global document mask; the CP member slices it to the local query interval. Two-dimensional padding
masks use one/True for valid tokens; fully masked query rows produce zero attention output before output bias.
CP is a training path and does not accept a KV cache; local attention retains cache support.

## MLA layouts and configuration

`MultiLatentAttention` and `GatedMultiLatentAttention` use the same family preparation path. The latter owns
NoPE and a mandatory output gate. Select a CP representation through the replacement request, for example:

```yaml
plan_overrides:
  - match: "model.layers.*.self_attn"
    replace:
      kind: module
      implementation:
        _target_: hyper_parallel.components.modules.MultiLatentAttention
        strategy: latent_kv_head
        overlap: true
```

An omitted `ulysses_degree` selects pure Ulysses. `ulysses_degree: 1` selects KV AllGather; another divisor
of the CP degree selects the hybrid. Ordinary MLA supports TP with pure Ulysses; its KV AllGather/hybrid
paths and gated MLA currently require TP1. `kv_exchange: all_to_all_v` requires query-partitioned CP.
Malformed options fail even when CP is inactive. Valid CP options can be retained on a local member and
checked against the actual topology when that member is later promoted.

Position handling remains owned by the mathematical member. Ordinary MLA uses the shared fused RoPE function
through `apply_position`. Passing `position_embeddings=None` skips rotation, but the ordinary factory does not
interpret `config.use_nope` as a NoPE switch. `GatedMultiLatentAttention` always uses NoPE, even when a caller
supplies position embeddings; it keeps the shared key channels and their checkpoint dimensions.

The low-level `MLAContextParallel(..., rotary_fn=...)` callback remains available. It receives a BSND query or
singleton-head shared key, local `(cos, sin)` embeddings (or `None`), and the `interleaved` keyword; the result
must preserve shape, dtype and device. Without an override, CP calls the member's `apply_position`, as local
execution does. The public `MultiLatentAttention` factory does not currently expose `rotary_fn`; an override
on a CP executor alone does not also change the local member's position math. The old `prepare_mla_cp`
helper was removed during the family migration, so its callback argument is not a drop-in family option.

The family declares the full parameter and I/O contract. For TP with SP, the entry is
`Shard(1) -> Replicate`, and the exit is `Partial -> Shard(1)`. Without SP the exit is
`Partial -> Replicate`. These transitions reuse `PrecompiledBoundary` and its existing collective lowering
or DTensor redistribution fallback. Output bias follows the original deferred-bias mechanism and is added
once after the boundary reduction. This follows the existing decoder-layer FSDP boundary; independent
FSDP wrapping of a biased output Linear is outside this change's validation scope.
The original projection object is retained. CP token ownership stays `Shard(1)` at both ends.
Ulysses temporarily redistributes sequence and heads inside attention; it does not persistently shard
weights on the CP axis.

`plan_overrides` still wins over family defaults, subject to the selected implementation's capabilities.
For example, override only `in_src` and `out_dst` to change external TP/SP token ownership, leaving the
family's computation layouts intact. Omitted parameter axes mean `Replicate`. Changing only the output
weight to `Replicate` while retaining head-sharded QKV is unsupported: it also requires a different
computation and communication plan. Parameter overrides replace the entire `params` field and must cover
all owned parameters. New or quantized projection parameters need an explicit family sharding role;
unknown parameter names are rejected rather than silently replicated.

## Attention execution structure and model coverage

Public names such as `GroupedQueryAttention` are family factories; the `Module` suffix identifies a concrete
`nn.Module` implementation. The suffix is a naming convention, not a requirement of family selection. The
family/CP construction paths consume replacement context; default mathematical members do not bind TP or CP resources.

MLA and GQA share family selection, final layout validation, TP/SP boundaries and `CPContext` transport.
Each local/CP family now shares its algorithm forward through class-defined stages:

- GQA: `prepare_attention_inputs` → `run_attention` → `restore_attention` → `project_output`.
  CP overrides input preparation and output restoration. It still consumes `projection_stages` and starts
  each ready transfer before later projections finish; hybrid K/V gathers remain ahead of the Q wait.
- MLA: `prepare_attention_inputs` → `local_attention` → `finish_attention`.
  `MLAContextParallel` prepares expanded/latent operands and their document metadata; it no longer calls FA
  or completes the layer. The inherited algorithm calls FA once. The CP output stage retains restoration,
  gate and output projection together so the existing paired backward schedule remains intact.

Per-call operand containers are temporary values, not module state or parameter wrappers. The mathematical
members continue to own projections, positions and FA. CP members only choose where those stages exchange
or restore ownership. Unlike the synchronous example, input preparation may interleave projection and
communication; moving every exchange after all projections would remove existing overlap windows.
Ordinary and gated members retain their current Parameter objects and checkpoint keys. Neither installs a
second instance-level CP forward wrapper; TP still uses the original framework boundary wrapper.

| Implementation | Mathematical/source contract | CP gate scheduling |
| --- | --- | --- |
| Ordinary GQA | Qwen3-style Q/K/V, optional Q/K Norm and RoPE; fused or separate projections | No output gate |
| Gated GQA | Qwen3.5 full attention: per-query-head Q/gate projection, Q/K Norm, partial RoPE, sigmoid output gate | Gate stays token-local; applied after output restoration, without a dedicated gate-overlap schedule |
| Gated MLA | Kimi NoPE MLA with an independent gate projection | Uses the existing MLA-specific gate and backward scheduling |

The Qwen3.5 text CP recipes in [qwen3_8](../../models/qwen3_8/recipes/train_online_p2p_kv_allgather.yaml) and
[qwen3_6_moe](../../models/qwen3_6_moe/recipes/train_online_p2p_kv_allgather.yaml) select gated GQA for full-attention
layers. Their Gated DeltaNet layers retain the existing GDN CP implementation. This migration does not cover every
Qwen3.5 VLM recipe or replace the full mixed model with GQA.

GQA accepts a global `[B, S]` padding mask. Query-partitioned CP must explicitly restore the local padded query rows
because gathered KV and local Q no longer have matching lengths. MLA CP currently accepts global Q/K visibility
masks for pure Ulysses, or causal document lengths for AG/hybrid; it does not yet accept this two-dimensional padding
mask contract. Shared FA support alone does not extend a model's CP input contract.

## Install custom operators

Some modules depend on the training operators from [Omni Ops](https://gitee.com/omniai/omni-ops). Ensure that CANN,
PyTorch and `torch_npu` are compatible before building the operators. MHC functions default to the CANN
ops-transformer operators instead (see `hyper_parallel/components/functional/README.md`); Omni Ops remains the
deprecated `backend="omni"` fallback for them and the operator source for other Omni-only paths.

```bash
git clone https://gitee.com/omniai/omni-ops.git
cd omni-ops
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

The `-c` option selects the target SoC. Omni Ops currently supports `ascend910b`, `ascend910_93`, and `ascend950`.
For example:

```bash
cd training/ascendc

# Select the command for the target device:
# Ascend 910B:   bash build.sh -c ascend910b
# Ascend 910_93: bash build.sh -c ascend910_93
# Ascend 950:    bash build.sh -c ascend950

# Example: build for Ascend 910_93
bash build.sh -c ascend910_93
```

Install the generated Ascend C operator package:

```bash
cd output

omni_operator_package="CANN-omni_training_custom_ops--linux.$(uname -m).run"
chmod +x "${omni_operator_package}"
"./${omni_operator_package}" \
  --quiet \
  --install-path="${ASCEND_HOME_PATH}/opp"

source "${ASCEND_HOME_PATH}/opp/vendors/omni_training_custom_transformer/bin/set_env.bash"
```

The PTA build script installs the wheel into the active Python environment. Activate the target environment first,
then build and install the wheel:

```bash
cd ../torch_ops_extension

which python3
which pip3
bash build_and_install.sh
```

Source the vendor environment in each new shell before using modules that depend on Omni Ops.

## Replace modules through YAML

Trainer applies module replacements declared in `plan_overrides` before model sharding and checkpoint loading. The
following example replaces Qwen3-VL-MoE Attention, Experts and RMSNorm modules:

```yaml
plan_overrides:
  - match: "*.language_model.layers.*.self_attn"
    replace:
      kind: module
      source_type: transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextAttention
      exact_type: true
      implementation:
        _target_: hyper_parallel.components.modules.GroupedQueryAttention

  - match: "*.language_model.layers.*.mlp.experts"
    replace:
      kind: module
      source_type: transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextExperts
      exact_type: true
      implementation:
        _target_: hyper_parallel.components.modules.GroupedExperts

  - match:
      - "*.language_model.layers.*.input_layernorm"
      - "*.language_model.layers.*.post_attention_layernorm"
      - "*.language_model.norm"
    replace:
      kind: module
      source_type: transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextRMSNorm
      exact_type: true
      implementation:
        _target_: hyper_parallel.components.modules.RMSNorm
```

- `match` specifies module FQN patterns from `model.named_modules()`. A list is supported when the same replacement
  applies to multiple paths.
- Optional `replace.source_type` checks the source module type. Set `replace.exact_type: true` to exclude subclasses.
- `replace.implementation._target_` specifies the high-performance replacement.

Invalid patterns, incompatible module types and unavailable weight conversions raise an error.

Replacements use `make_transforms()` only when parameter names or layouts change. Otherwise, they reuse the source
parameters directly.

## Run training

Add `plan_overrides` to the training YAML and run the corresponding Trainer entry point. For VLM training:

```bash
source "${ASCEND_HOME_PATH}/opp/vendors/omni_training_custom_transformer/bin/set_env.bash"

torchrun \
  --nproc_per_node=8 \
  --module examples.training_demo.train_vlm \
  path/to/train.yaml
```

Run one complete training step first to verify checkpoint loading, forward and backward. For performance comparisons,
keep the training configuration unchanged and only add or remove `plan_overrides`.
