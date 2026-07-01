# GLM5 training example

This directory provides the GLM5 Trainer entry point for `scripts/train_lm.py`.
It follows the same example layout as the existing Qwen examples: one user-facing
`train.yaml` for GLM5 training.

## Scope

The GLM5 Trainer path supports:

- dense GLM5 causal LM forward/loss/backward training;
- checkpoint save, load, and resume with model, optimizer, scheduler, RNG, and
  dataloader state;
- MLA attention and 2D/4D attention mask handling;
- MoE construction and EP=2 expert dispatch;
- DSA sparse-attention construction and CP=2 context-parallel training.

Tensor parallelism is not implemented for GLM5. Setting
`train.accelerator.tp > 1` raises `NotImplementedError`.

Cached decode supports append-only autoregressive decoding with contiguous
history positions. Packed sequences, non-contiguous cached positions, cache
reuse, and prefix stitching are not part of this training example.

## Configuration

`train.yaml` is the only training configuration committed under
`examples/glm5`. It is intended as the stable user-facing entry point.
Validation scenarios such as MoE, EP, CP, DSA, scaled architecture alignment,
and cross-framework comparison are reproduced with command-line overrides and
external validation materials, rather than additional YAML files under
`examples`.

## Unit Tests

GLM5 Trainer tests live under `tests/torch/trainer/test_glm5_trainer.py`.
Trainer callback configuration regression tests live under `tests/ut/trainer`.
Together they cover model discovery, GLM5 batch preparation through
`BaseTrainer`, shifted CausalLM loss semantics, CP batch sharding, checkpoint
callback round-trip, parallelization guards, and nested train logging/checkpoint
configuration.

```bash
python -m pytest \
  tests/torch/trainer/test_glm5_trainer.py \
  tests/ut/trainer/test_checkpoint_callback_config.py \
  tests/ut/trainer/test_logging_callback_config.py -q
```

Validated result:

```text
13 passed
```

## Cross-Framework Validation

The full official GLM5 checkpoint is too large for the minimum validation
environment, so cross-framework checks use scaled GLM5 dense and MoE+MLA+DSA
variants. The variants keep GLM5 module signatures and the GLM-5 tokenizer
vocabulary size while reducing layer width, layer count, and expert count.

The validation compares the same fixed batch and same exported weights. The
Transformers side uses the official `GlmMoeDsaConfig` and
`GlmMoeDsaForCausalLM` classes; it does not import
`hyper_parallel.models.glm5`.

- Transformers official `GlmMoeDsaForCausalLM` vs HyperParallel GLM5;
- LLaMAFactory `CustomSeq2SeqTrainer.compute_loss` vs HyperParallel GLM5.

These checks validate single-card loss/logits semantics for the GLM5 model
structure paths covered by Dense, MoE, MLA, and the official DSA indexer. CP and
EP are HyperParallel parallel-training strategies and are validated by
in-repository tests and separate ST validation materials instead of
external-framework comparison.

## Verified Result

The following results were collected on Ascend NPU with GLM5 validation commands
based on `train.yaml` and fixed validation materials.

```text
Scaled dense GLM5 fp32 1c vs DP2, 100 steps:
common_steps: 100
avg_diff: 6.55615000e-05
max_diff: 2.77690000e-04
pass_avg_5e-3: True

Scaled MoE+MLA+DSA GLM5 fp32 1c vs DP2, 100 steps:
common_steps: 100
avg_diff: 6.64436000e-05
max_diff: 2.55470000e-04
pass_avg_5e-3: True

MoE preset fp32 1c vs EP2, 100 steps:
avg_diff: 4.83024000e-05
max_diff: 2.59350000e-04
pass_avg_5e-3: True

DSA preset fp32 1c vs CP2, 100 steps:
avg_diff: 3.80690000e-05
max_diff: 2.98080000e-04
pass_avg_5e-3: True

Transformers scaled dense:
hf_loss: 12.262624740600586
hyper_loss: 12.262624740600586
logits_max_abs_diff: 0.0
loss_abs_diff: 0.0
pass_logits: True
pass_loss: True

Transformers scaled MoE+MLA+DSA:
hf_loss: 12.128127098083496
hyper_loss: 12.128125190734863
logits_max_abs_diff: 0.0
loss_abs_diff: 1.9073486328125e-06
pass_logits: True
pass_loss: True

LLaMAFactory scaled dense:
llamafactory_loss: 12.262624740600586
hyper_loss: 12.262624740600586
logits_max_abs_diff: 0.0
loss_abs_diff: 0.0
pass_logits: True
pass_loss: True

LLaMAFactory scaled MoE+MLA+DSA:
llamafactory_loss: 12.128129005432129
hyper_loss: 12.128125190734863
logits_max_abs_diff: 0.0
loss_abs_diff: 3.814697265625e-06
pass_logits: True
pass_loss: True

Cross-framework state_dict diagnostics:
missing_hp_keys: []
unexpected_hf_keys: []
shape_mismatch_keys: {}
```

The fp32 settings are validation-only overrides. They remove low-precision
rounding from strict alignment checks and do not change the normal mixed
precision training path exposed by `train.yaml`.
