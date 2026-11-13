# Qwen3-Omni Model-Family Integration

**English** | [简体中文](README.zh.md)

Qwen3-Omni is Alibaba's any-to-any multimodal MoE model (text / image / audio / video input, with an
optional speech-generation talker). Its thinker nests a vision tower, an audio tower and a sparse-MoE
decoder; audio output is disabled for training (`enable_audio_output=false`), leaving the thinker as
the trainable module. This directory contains HyperParallel's integration for this family:
stock-modeling fixes, derived-buffer materialization, vision FSDP2 unit declarations, the audio/video
data transformation, and training recipes.

## Registration Identity

| Item | Value |
|---|---|
| HyperParallel family name (`models/<family>/` directory) | `qwen3_omni` |
| HF `config.model_type` | `qwen3_omni_moe` |
| HF `architectures` | `Qwen3OmniMoeForConditionalGeneration` |
| Implementation source | Native Transformers implementation (`transformers.models.qwen3_omni_moe`), `trust_remote_code: false` |

The HF checkpoint's `model_type` carries the MoE suffix the family directory does not;
`hyper_parallel/models/registry.py` maps it through `_FAMILY_DIR_ALIASES`
(`qwen3omnimoe → qwen3_omni`). `get_model_adapter()` resolves all three spellings:
`qwen3_omni`, `qwen3_omni_moe`, and `Qwen3OmniMoeForConditionalGeneration`.

## Directory Structure

```text
models/qwen3_omni/
├── __init__.py                      # get_adapter_spec() entry point
├── adapter/
│   ├── __init__.py                  # QWEN3_OMNI_MODEL_TYPES: model_type spellings accepted by runtime guards
│   ├── model_patches.py             # Stock-Transformers fixes (training forward, packed audio, patch-embed
│   │                                #   Linear routing, deepstack 2-D masks, optional fused NPU MoE)
│   ├── materialization.py           # Rebuild recipes for derived positional buffers (rotary inv_freq,
│   │                                #   audio sinusoids) — stock CPU fp32 values, declared before sharding
│   ├── omni_randinit.py             # Random-init builder (HyperAutoModelForMultimodalLM.from_config)
│   ├── omni_moe_fused_npu.py        # Optional NPU fused MoE kernels (env HYPER_OMNI_MOE_FUSED_NPU=1)
│   ├── omni_moe_packed_experts.py   # Optional NPU packed-expert MoE path
│   ├── data/
│   │   └── transform_fn.py          # Qwen3OmniTransform: audio/video/image SFT record transform
│   └── registration.py              # ModelAdapterSpec: materialization + stock_model_patches + fsdp_wrap_modules
└── recipes/
    └── train.yaml                   # Single-node 8-device SFT recipe
```

## Integration Details

- **Stock-model patches** (`adapter/model_patches.py`, declared through the
  `stock_model_patches` provider): the installed Transformers ships defects that break
  Qwen3-Omni training — generation-only wrappers without a training `forward`, a
  packed-audio dead branch, a patch-embed Conv3d whose NPU backward segfaults (routed
  through an equivalent Linear — bit-identical forward, same weight storage gradient),
  and a deepstack 3-D-mask joint-branch crash. Every patch guards on the buggy stock
  pattern, so an already-fixed modeling is left untouched, and the set is idempotent.
- **Derived-buffer materialization** (`adapter/materialization.py`, declared through the
  `materialization` provider): `to_empty()` erases non-persistent buffers the checkpoint
  does not store (thinker/vision rotary `inv_freq`, audio sinusoid
  `positional_embedding`). The provider registers rebuild recipes capturing the exact
  stock `__init__` CPU fp32 values; the generic dtype path restores fp32-sourced
  rebuildable buffers after the `model_init_dtype` cast, preserving byte-parity with a
  plain `from_pretrained` model (stock Transformers computes them with explicit
  `.float()`).
- **Vision FSDP2 units** (`fsdp_wrap_modules` provider): the omni `visual` branch lives
  outside the HF decoder-container convention; `patch_embed` and `merger` are declared
  as explicit FSDP2 child units so their all-gather hooks fire on their own `__call__`.
- **Data transformation** (`adapter/data/transform_fn.py`, `Qwen3OmniTransform` +
  `build_qwen3_omni_transform`): audio/video/image SFT records through the shared VLM
  dataset pipeline (`hyper_parallel.data.vlm`); the recipe references it directly
  through `dataset.data_transform._target_`, so registration needs no changes.
- **Compile containers**: the family declares `get_compile_layers()` on the model class
  (`thinker.model.layers` / `talker.model.layers`), satisfying the model-owned contract
  in `distributed.compile`.

## Recipes

See [recipes/train.yaml](recipes/train.yaml) for the single-node eight-device SFT recipe.
An end-to-end ScienceQA SFT training example lives in
[examples/qwen3_omni_sft](../../../examples/qwen3_omni_sft).
