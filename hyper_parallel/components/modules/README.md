# High-performance modules

`hyper_parallel.components.modules` provides reusable model building blocks and high-performance modules for Ascend NPU.
Hardware-backed modules can be used directly in Python or replace existing model modules through Trainer YAML.
Portable routing and residual components also support reference execution without an NPU backend.

## Available modules

| Category | Modules |
| --- | --- |
| Attention | `GQAAttention`, `GatedGQAAttention`, `MLAAttention`, `DSAAttention`, `DeepseekV32DSAAttention` |
| MoE and MLP | `GroupedExperts`, `SharedExpert`, `SwiGLUMLP`, `LatentMoE`, `QuantileBalancer` |
| Normalization | `RMSNorm`, `OffsetRMSNorm` |
| MHC | `MhcPreModule`, `MhcPostModule` |

Reusable high-performance functions are provided separately in [`hyper_parallel.components.functional`](../functional/README.md).

Modules can also be constructed directly:

```python
from hyper_parallel.components.modules import RMSNorm

norm = RMSNorm(hidden_size=4096, eps=1e-6).npu()
output = norm(hidden_states)
```

## Reusable routing and residual components

`LatentMoE` routes full-width hidden states, evaluates experts in an optional latent space,
combines their outputs in FP32, then applies the latent norm and output projection.
Its optional shared expert consumes the original full-width input. The calling model supplies
executable router/expert/projection modules and retains ownership of parameter names and initialization.
The portable reference does not require an NPU backend.

`QuantileBalancer` records local required-bias histograms from sigmoid scores and the router's
existing TopK indices. `finalize_step(bias)` commits a local step. Distributed callers use
`finalize_quantile_balancers` from `distributed.expert_parallel.balancing` with explicitly declared,
non-overlapping token groups. Commit only after a successful optimizer update; discard observations
on skipped updates. Evaluation and recomputation must not record observations. Visual routing masks
are independent of language-loss masks. The existing router finalize/discard callback protocol owns
this lifecycle; no model-specific Trainer is needed.

`expert_compute.py` provides `grouped_expert_mlp`, `grouped_expert_slots` and
`grouped_routed_experts`. Their activation callable consumes complete concatenated gate/up rows,
so SiTU and SwiGLU can share the compute path without changing either activation's mathematics.
Slots return unweighted outputs; routed tokens apply FP32 probabilities after expert computation.
The grouped operations require their existing NPU primitives, imported only when used.
`activation` and `fc1_activation` checkpoint scopes replay local expert computation only.
`GroupedExperts` uses this common bias-free GLU computation and preserves its existing biased and
pre-FC2-weighting paths. Its optional `activation` constructor argument binds a complete GLU.

`attention_residuals.py` contains explicit block-history materialization and boundary operations.
The model chooses block boundaries and passes each history Tensor through checkpoint/swap inputs.
Each consumer materializes its own history to preserve the original gradient accumulation grouping.
The stateless eager/fused attention-residual operation remains in `components.functional`.

## Module families

Registered module families select an implementation from explicit configuration rules before model construction.
See [Module Family](README.module_family.md) for GroupedExperts MXFP8 examples, rule semantics, and extension guidance.

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
        _target_: hyper_parallel.components.modules.GQAAttention

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
