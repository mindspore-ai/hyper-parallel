# High-performance modules

`hyper_models.modules` provides drop-in high-performance modules for Ascend NPU. Modules can be used directly in
Python or declaratively replace existing model modules through Trainer YAML.

## Available modules

| Category | Modules |
| --- | --- |
| Attention | `GQAAttention`, `GatedGQAAttention`, `MLAAttention`, `DSAAttention`, `DeepseekV32DSAAttention` |
| MoE and MLP | `GroupedExperts`, `SharedExpert`, `SwiGLUMLP` |
| Normalization | `RMSNorm`, `OffsetRMSNorm` |
| MHC | `MhcPreModule`, `MhcPostModule`, `MhcPostProcessModule` |

Reusable high-performance functions are provided separately in [`hyper_models.ops`](../ops/README.md).

## Install custom operators

Some modules depend on the training operators from
[Omni Ops](https://gitee.com/omniai/omni-ops). Ensure that CANN, PyTorch and `torch_npu` are compatible before
building the operators.

```bash
git clone https://gitee.com/omniai/omni-ops.git
cd omni-ops
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Build and install the AscendC operator package. Replace `ascend910_93` with the target SoC when needed.

```bash
cd training/ascendc
bash build.sh -c ascend910_93

cd output
chmod +x CANN-omni_training_custom_ops--linux.<arch>.run
./CANN-omni_training_custom_ops--linux.<arch>.run \
  --quiet \
  --install-path=${ASCEND_HOME_PATH}/opp

source ${ASCEND_HOME_PATH}/opp/vendors/omni_training_custom_transformer/bin/set_env.bash
```

Build and install the PyTorch Adapter wheel:

```bash
cd ../torch_ops_extension
bash build_and_install.sh
```

The vendor environment must be sourced again in each new shell before running modules that use Omni Ops.

## Replace modules through YAML

Trainer applies module replacements declared in `plan_overrides` before model sharding and checkpoint loading. The
following example replaces Qwen3-VL-MoE Attention, Experts and RMSNorm modules:

```yaml
plan_overrides:
  - match: "*.language_model.layers.*.self_attn"
    module_type: transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextAttention
    exact_type: true
    replace_module:
      _target_: hyper_models.modules.GQAAttention

  - match: "*.language_model.layers.*.mlp.experts"
    module_type: transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextExperts
    exact_type: true
    replace_module:
      _target_: hyper_models.modules.GroupedExperts

  - match:
      - "*.language_model.layers.*.input_layernorm"
      - "*.language_model.layers.*.post_attention_layernorm"
      - "*.language_model.norm"
    module_type: transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextRMSNorm
    exact_type: true
    replace_module:
      _target_: hyper_models.modules.RMSNorm
```

- `match` is a module FQN or glob pattern. A list can be used when the same replacement applies to several paths.
- `module_type` is the import path of the source module and prevents unintended replacements.
- `exact_type: true` matches only the declared type rather than its subclasses.
- `replace_module._target_` selects the high-performance replacement.

Use names from `model.named_modules()` when adapting a new model. Trainer reports an error when a pattern matches no
module, the source type is incompatible, or required weight conversion cannot be applied.

Modules that change checkpoint layout declare reversible conversions through `make_transforms()`. Modules whose
layout is unchanged reuse the source parameters directly.

## Run training

Use the regular Trainer entry point and pass the YAML containing `plan_overrides`:

```bash
source ${ASCEND_HOME_PATH}/opp/vendors/omni_training_custom_transformer/bin/set_env.bash

torchrun \
  --nproc_per_node=8 \
  --module examples.training_demo.train_vlm \
  path/to/train.yaml
```

Before a performance run, use a short sequence to verify checkpoint loading, forward, backward and one optimizer
step. Keep all other training settings unchanged when comparing the baseline and replacement configurations.
