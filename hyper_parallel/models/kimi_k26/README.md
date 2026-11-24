# Kimi-K2.6 Model Integration

**English** | [简体中文](README.zh.md)

Kimi-K2.6 uses the native Transformers `kimi_k25` configuration and source implementation;
the HyperParallel model identity is `kimi_k26`. Use the native processor and
`trust_remote_code: false`. Legacy remote-code `auto_map` entries in published assets do not
require executing remote model code.

## Files and Responsibilities

```text
models/kimi_k26/
├── __init__.py                         # Lightweight package entry
├── kimi_k26_spec.py                    # Identity, public families, FSDP boundaries, finite legacy YAML targets
├── modeling_kimi_k26.py                # Native model composition; decoder-owned CP input and chunked loss
├── dataset_adapter.py                 # Full media inputs and globally shifted local CP targets
├── dataset_preprocessing/
│   ├── __init__.py                     # Preprocessing package entry
│   └── transform_fn.py                 # Processor output to Omni eager/packing samples
├── validation/
│   ├── __init__.py                     # Validation helper package entry
│   └── cropped_model.py                # Explicit validation crop/freeze builder, without public algorithms
├── generated/                         # Optional codegen output
└── recipes/
    ├── train.yaml                     # Existing released-model training configuration
    ├── run_kimi_k26_4sn.sh             # Existing multi-node launcher
    └── prepare_kimi_vlm_data.py        # Demonstration image/text data generator
```

## Module Families and Architecture

- The root spec selects public `MultiLatentAttention`, `GroupedExperts`, `SwiGLUMLP`, `RMSNorm`,
  `RotaryEmbedding`, `LearnedAxialVisionTransformer`, `SpatialPatchProjection`, and `CausalLMHead`.
  These own source-state adoption, local weight conversion, mathematics and parallel execution.
  No duplicate private family is needed for this model.
- Native model code owns visual/language fusion, source output protocols and terminal loss composition.
  No model or visual instance receives an injected `forward`.
- The decoder partitions fully fused `inputs_embeds`, preserves explicit global position offsets/resets,
  and constructs the corresponding CP mask. The complete Omni CP pipeline requires separate acceptance;
  CP1 results do not qualify it.
- Source routing and shared-expert semantics are preserved; Experts own GEMM and EP communication.
  Historical benchmark EP targets resolve through finite root-spec aliases to the public recipe.
- Independent matrix boundaries are retained for Muon. Random-initialization validation cases live outside
  this repository; existing recipes are preserved. See the [validation report](../../../docs/model-refactor-validation-report.md).

## Using the Recipe

`recipes/train.yaml` targets the **main scale of 4 SuperPoDs / 512 devices**, with
`cp 1 × dp_shard 512 (= edp 4 × ep 128)`, `GBS 512 = 512 × MBS 1`, and sequence length 8192.
The previous balanced-routing benchmark at this topology measured a minimum step time of **18.2617 s**,
padded throughput of **228,873 tok/s**, and peak allocated memory of **39.19 GB** without activation swap.

```bash
# Run once on each node, with NODE_RANK from 0 to NNODES-1.
NNODES=64 NODE_RANK=0 MASTER_ADDR=<rank0-ip> MASTER_PORT=6100 \
    bash hyper_parallel/models/kimi_k26/recipes/run_kimi_k26_4sn.sh /path/to/Kimi-K2.6
```

Prerequisites and notes:

- A Kimi-K2.6 multimodal checkpoint is required, with configuration, processor, and weights in one directory. For local/offline use, the launcher
  sets `model.*` and `dataset.model_assets.*` to that directory through dotted overrides and enables
  `local_files_only=true`. Recipe defaults point to the public `moonshotai/Kimi-K2.6`.
- Keep `trust_remote_code` set to `false` for both model and `dataset.model_assets`, using Transformers'
  native `kimi_k25` implementation. The VLM Trainer forwards `model.trust_remote_code` to `AutoProcessor`.
- `recipes/prepare_kimi_vlm_data.py` generates demo data: eight records by default, with 1024 used by the launcher.
  Override `--dataset.data_path` for real data such as COCO.
- Other scales (1SN/2SN/3SN) require coordinated changes to `accelerator.ep_size` and `fsdp_config`'s
  `dp_shard_size` / `edp_shard_size`. By default, the launcher rejects world sizes other than 512.
