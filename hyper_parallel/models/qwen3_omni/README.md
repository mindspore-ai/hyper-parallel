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

## Known Hardware Validation Boundaries (Single-Node 910B2 Measurements, 2026-11)

- **The stock patch-embed `Conv3d` segfaults in NPU backward and is routed through an equivalent
  `Linear`**: the family patch rewrites the vision patch embedding to a mathematically identical
  linear formulation (bit-identical forward, same weight-storage gradient). The stock path must not
  be restored on Ascend; the patch guards on the buggy stock pattern and leaves an already-fixed
  modeling untouched.
- **CPU offload is required at 30B scale on 64 GB devices**: with FSDP2 full sharding over 8 devices
  but no offload, bf16 parameters, gradients and fp32 AdamW states together with activations exceed
  the 64 GB device budget. The recipe enables `fsdp_config.enable_offload: true`; measurements on
  8 × Ascend 910B2 (64 GB) show a peak of roughly 9 GB per device with offload, versus OOM without.
- **`max_seq_len` is bounded by memory, not correctness**: the validated SFT runs used
  `max_seq_len: 1024` with `micro_batch_size: 1` / `global_batch_size: 8`. Longer sequences are
  expected to fit at this scale but are not validated; raise before enlarging batch.
- **The optional fused / packed NPU MoE paths are off by default**: they change the expert compute
  path (grouped-GEMM / packed experts) and are gated behind `HYPER_OMNI_MOE_FUSED_NPU=1`. The
  default eager path is the numerically validated configuration; the packed path was validated for
  step-time (loss parity at step 0 bit-identical), not for end-to-end convergence.

## Dependencies and Asset Prerequisites

- **Transformers supplies the native model architecture; this family ships no modeling code**.
  Construction resolves `Qwen3OmniMoeForConditionalGeneration` from Transformers'
  native `qwen3_omni_moe` implementation with `trust_remote_code: false`; the checkpoint's
  `auto_map` remote code is deliberately not used. Transformers must provide native
  `qwen3_omni_moe` support.
- **Weights must be BF16**: the integration is validated against the official
  `Qwen/Qwen3-Omni-30B-A3B-Instruct` BF16 release. Loading is strict — key or shape mismatches
  raise errors, which also validates at runtime that the native implementation matches the
  official weight structure.
- **Audio output is disabled for training** (`enable_audio_output: false`): only the thinker
  (vision tower + audio tower + sparse-MoE decoder) is trainable. The talker / code2wav speech
  generation stack is inference-only in this integration.
- The recipe defaults to public Hugging Face resources (`Qwen/Qwen3-Omni-30B-A3B-Instruct`);
  local or offline checkpoints and datasets are passed through dotted overrides on the command
  line, as the launcher does.

## Using the Recipe

`recipes/train.yaml` targets a **single node with 8 Ascend NPUs**: FSDP2 `dp_shard_size: 8`
(`tp = cp = ep = pp = 1`), `GBS 8 = 8 × MBS 1`, `max_seq_len: 1024`, bf16, AdamW 1e-5 cosine.
The end-to-end ScienceQA SFT example bundles the same recipe with a launcher:

```bash
# Single node, 8 devices. Local or offline resources are set through env vars.
MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct \
DATA_PATH=/path/to/scienceqa/train.json \
CKPT_DIR=/path/to/checkpoints \
    bash examples/qwen3_omni_sft/run_train.sh
```

Prerequisites and notes:

- A Qwen3-Omni-30B-A3B BF16 checkpoint (configuration, processor and weights in one directory)
  and a ScienceQA-format annotation file are required. `MODEL_PATH` / `DATA_PATH` / `CKPT_DIR`
  override the recipe defaults without editing the yaml; `N_EPOCHS` scales
  `training.train_iters` (`256 × N_EPOCHS` for the 2048-record subset the launcher comment
  assumes).
- Keep `trust_remote_code` at `false` — the family targets the native Transformers
  implementation.
- The VLM data pipeline requires `tp = cp = pp = 1`; only FSDP2 sharding varies the parallel
  plan at this scale.

## Tests

| Test | Coverage |
|---|---|
| `tests/ut/auto_models/models/qwen3_omni/test_qwen3_omni_adapter_contracts.py` | Adapter contract: identity resolution for all three spellings, provider declaration, vision FSDP2 unit naming, patch idempotency, fp32 buffer restoration across dtype casts |

## Recipes

See [recipes/train.yaml](recipes/train.yaml) for the single-node eight-device SFT recipe.
An end-to-end ScienceQA SFT training example lives in
[examples/qwen3_omni_sft](../../../examples/qwen3_omni_sft).
