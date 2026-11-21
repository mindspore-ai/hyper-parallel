# High-performance modules

`hyper_parallel.components.modules` provides drop-in high-performance modules for Ascend NPU. Modules can be used directly in
Python or declaratively replace existing model modules through Trainer YAML.

## Available modules

| Category | Modules |
| --- | --- |
| Attention | `GroupedQueryAttention`, `GatedGroupedQueryAttention`, `MultiLatentAttention`, `GatedMultiLatentAttention`, `DeepseekSparseAttention` |
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
The public [GroupedExperts family](moe/grouped_experts/README.md) selects local or EP execution.
See [Module Family](README.module_family.md) for legacy MXFP8 examples, rule semantics, and extension guidance.

For attention replacement parameters, topology settings and framework-injected arguments, see
[Attention YAML configuration](attention/readme.md). The
[DeepSeek Sparse Attention guide](attention/deepseek_sparse_attention/readme.md) explains native `config=`
construction, source-module replacement, and CP / TP support.

## Attention families

Choose an attention family for the model's mathematical variant:

| Family | Features and guide |
| --- | --- |
| [GroupedQueryAttention](attention/grouped_query_attention/README.md) | GQA with fused or separate QKV; Ulysses, KV AllGather and hybrid CP with input overlap |
| [GatedGroupedQueryAttention](attention/gated_grouped_query_attention/README.md) | Qwen3.5 full attention with RoPE and a sigmoid output gate |
| [MultiLatentAttention](attention/multi_latent_attention/README.md) | MLA with expanded or latent KV communication and supported forward/backward overlap |
| [GatedMultiLatentAttention](attention/gated_multi_latent_attention/README.md) | NoPE MLA with an independent output gate and gate-aware CP scheduling |
| [DeepseekSparseAttention](attention/deepseek_sparse_attention/readme.md) | DeepSeek-V3.2 sparse attention, native construction and KV-AllGather CP |

Each guide includes a YAML fragment, configuration choices, file responsibilities, the execution flow and support limits.
Families support direct `config=` construction or `module=` conversion from a compatible source.
Trainer YAML selects the family through `replace.implementation._target_`; `accelerator.cp_size` selects its local or CP member.

```text
YAML / config → family selection and layout → framework binding → shared attention forward
```

The algorithm owns the math and shared forward; the CP member handles input exchange and output restoration.
TP/SP uses the framework's existing layout redistribution. Model/runtime inputs must provide the position information
and document boundaries required by the selected attention family. Full training recipes are linked from the guides.

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

Replacements declare checkpoint conversion through `make_transforms()` and can retain source parameter identity.
A replacement with a custom `reset_parameters()` must also declare `make_transforms()`, returning an empty list
when no conversion is needed. The builder then arranges its reset after meta parameters are materialized.

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
