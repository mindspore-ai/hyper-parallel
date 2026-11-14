# High-performance modules

`hyper_parallel.components.modules` provides drop-in high-performance modules for Ascend NPU. Modules can be used directly in
Python or declaratively replace existing model modules through Trainer YAML.

## Available modules

| Category | Modules |
| --- | --- |
| Attention | `GroupedQueryAttention`, `SplitGroupedQueryAttention`, `GatedGroupedQueryAttention`, `MultiLatentAttention`, `GatedMultiLatentAttention`, `DSAAttention`, `DeepseekV32DSAAttention` |
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
- `SplitGroupedQueryAttention` retains independent Q/K/V projections and their checkpoint names. This preserves
  the early communication windows of the former Qwen3-MoE asynchronous implementation.
- `GatedGroupedQueryAttention` uses the per-query-head sigmoid output gate of Qwen3.5 full attention.
  This does not replace the model's linear-attention layers.

`overlap` changes launch/wait placement within the chosen representation; it does not change projection weights.
The CP members reuse `CPContext`, while `TensorParallelExecution` owns SP input gathering and TP output reduction.
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

There is no additional CP forward-wrapper replacement. Use `SplitGroupedQueryAttention` with `overlap: true`
to retain separate projection calls while overlapping ready Q/K/V transfers.

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

## Install custom operators

Some modules depend on the training operators from [Omni Ops](https://gitee.com/omniai/omni-ops). Ensure that CANN,
PyTorch and `torch_npu` are compatible before building the operators.

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
