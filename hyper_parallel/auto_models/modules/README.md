# High-performance modules

`hyper_parallel.auto_models.modules` provides drop-in high-performance modules for Ascend NPU. Modules can be used directly in
Python or declaratively replace existing model modules through Trainer YAML.

## Available modules

| Category | Modules |
| --- | --- |
| Attention | `GQAAttention`, `GatedGQAAttention`, `MLAAttention`, `DSAAttention`, `DeepseekV32DSAAttention` |
| MoE and MLP | `GroupedExperts`, `SharedExpert`, `SwiGLUMLP` |
| Normalization | `RMSNorm`, `OffsetRMSNorm` |
| MHC | `MhcPreModule`, `MhcPostModule`, `MhcPostProcessModule` |

Reusable high-performance functions are provided separately in [`hyper_parallel.auto_models.ops`](../ops/README.md).

Modules can also be constructed directly:

```python
from hyper_parallel.auto_models.modules import RMSNorm

norm = RMSNorm(hidden_size=4096, eps=1e-6).npu()
output = norm(hidden_states)
```

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

Install the generated AscendC operator package:

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
    module_type: transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextAttention
    exact_type: true
    replace_module:
      _target_: hyper_parallel.auto_models.modules.GQAAttention

  - match: "*.language_model.layers.*.mlp.experts"
    module_type: transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextExperts
    exact_type: true
    replace_module:
      _target_: hyper_parallel.auto_models.modules.GroupedExperts

  - match:
      - "*.language_model.layers.*.input_layernorm"
      - "*.language_model.layers.*.post_attention_layernorm"
      - "*.language_model.norm"
    module_type: transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextRMSNorm
    exact_type: true
    replace_module:
      _target_: hyper_parallel.auto_models.modules.RMSNorm
```

- `match` specifies module FQN patterns from `model.named_modules()`. A list is supported when the same replacement
  applies to multiple paths.
- `module_type` specifies the source module type. Set `exact_type: true` to exclude subclasses.
- `replace_module._target_` specifies the high-performance replacement.

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
