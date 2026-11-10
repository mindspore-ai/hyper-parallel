# Kimi-K2.6 Model-Family Integration

**English** | [简体中文](README.zh.md)

Kimi-K2.6 is Moonshot's multimodal (image-text) MoE model. Its text tower uses a DeepSeek-V3-style MLA + MoE architecture,
with a separate vision tower and projector. This directory contains HyperParallel's integration for this family: module replacement factories, CP/sharding rules,
loss adapters, data transformation entry points, and training recipes.

## Registration Identity

| Item | Value |
|---|---|
| HyperParallel family name (`models/<family>/` directory and registration identity) | `kimi_k26` |
| HF `config.model_type` (actual K2.6 checkpoint value) | `kimi_k25` |
| HF `architectures` | `KimiK25ForConditionalGeneration` |
| Implementation source | Native Transformers implementation (`transformers.models.kimi_k25`), `trust_remote_code: false` |

K2.6 checkpoints retain K2.5's native implementation and configuration names. The directory and registration identity therefore use `kimi_k26` to match the
user-facing product version, while `hyper_parallel/models/registry.py` maps the HF spelling `kimi_k25` through
`_FAMILY_DIR_ALIASES` (`kimik25 → kimi_k26`). `get_model_adapter()` resolves all three spellings:
`kimi_k26`, `kimi_k25`, and `KimiK25ForConditionalGeneration`.

## Directory Structure

```text
models/kimi_k26/
├── __init__.py                      # get_adapter_spec() entry point
├── adapter/
│   ├── __init__.py                  # KIMI_MODEL_TYPES: model_type spellings accepted by runtime guards
│   ├── chunk_loss.py                # Training Chunk Loss forward replacement, bound to model.forward
│   ├── distributed/
│   │   └── context_parallel.py      # CP input sharding: inputs_embeds windows, global position_ids, offset masks
│   ├── data/
│   │   └── transform_fn.py          # Omni transforms using native image_grid_thw: eager and packing
│   ├── validation/
│   │   └── cropped_model.py         # build_cropped_kimi_vlm model builder (recipe model._target_)
│   └── registration.py              # ModelAdapterSpec: sharding_rules + loss
└── recipes/
    ├── train.yaml                   # Main recipe: 4 SuperPoDs / 512 devices, best measured configuration
    ├── run_kimi_k26_4sn.sh          # Multinode launcher: 64 nodes × 8 devices by default
    └── prepare_kimi_vlm_data.py     # Generate demo image-text data for offline self-tests
```

Data transformations also reside in this family: `adapter/data/transform_fn.py` (`KimiVLMChatTransform` /
`KimiOmniTransform` / `KimiPackingOmniTransform` plus two builders), following the placement used by
`models/deepseek_v41/adapter/data/`. The recipe references them directly through
`dataset.data_transform._target_`; `ModelAdapterSpec` does not register data adapters, so registration needs no changes.

## Integration Details

- **Sharding rules** (`adapter/registration.py`): The nested text tower uses MLA, but low-rank down projections
  (`q_a_proj` / `kv_a_proj_with_mqa`) remain replicated, without sharding the LoRA rank dimension. Up projections
  (`q_b_proj` / `kv_b_proj`) are column-sharded along heads. The vision tower and `mm_projector` remain fully replicated: their native attention
  rearranges heads over the full hidden dimension and cannot compose with TP sharding. The DeepSeek-V3 MLA naming rules therefore cannot be applied directly.
- **Chunk Loss** (`adapter/chunk_loss.py`): An atomic forward replacement binds chunked cross-entropy to
  `model.forward`, avoiding resident full-sequence logits. It activates only when `chunk_loss_targets` is present.
  Currently requires `tp_size=1`: the core `ChunkedCausalLMLoss` rejects chunk loss with vocabulary sharding (TP)
  during `bind_model`. The recipe accordingly uses only preshifted targets: Omni batches default
  `labels_are_shifted` to `True`, and their published `shift_labels` are the required chunk-loss targets.
- **CP input sharding** (`adapter/distributed/context_parallel.py`): The dataloader partitions samples by `dp_rank`.
  CP ranks in the same DP group receive identical samples and each run the vision tower. Before the text tower, `inputs_embeds` is sliced into
  `[cp_rank*L, (cp_rank+1)*L)` windows. RoPE `position_ids` are reconstructed from the global window, and a 4-D offset
  causal-plus-padding mask is supplied. This is a no-op when `cp_size <= 1`, so it can be called unconditionally.
- **Model construction** (`adapter/validation/cropped_model.py`): Check that `config.model_type` belongs to
  `KIMI_MODEL_TYPES`, truncate text-tower layer/expert counts, then parallelize through HyperParallel's model builder. Supports
  `freeze_patterns` before plan/FSDP inference; freezing the vision tower also removes its backward computation and activations.
- **EP compute factory** (`adapter/distributed/ep_compute.py`): The family composes its own
  `deepseekv3_sigmoid_group_shared` prototype, colocated as in `models/qwen3_moe/adapter/distributed/`,
  through the public `build_ep_compute` interface, requiring no framework changes. `fix_router: true` selects the framework's
  **balanced routing** (`routing.make_balanced_router`, resolved by the shared skeleton): measured token skew with real routing loads the busiest rank to 6.3 times the mean,
  a major cause of OOM. Balanced routing gives every destination rank exactly the same number of slots,
  allowing comparable benchmark measurements to stabilize within 2–3 steps.
  ⚠️ This mode is **benchmark-only**: gate scores are ignored, and `loss`, `grad_norm`, and expert-hit distributions are meaningless.
  Set it to `false` for correctness and convergence validation, restoring the framework's real sigmoid-group routing.
- **Data transformation** (`adapter/data/transform_fn.py`): Uses the native processor's
  `image_grid_thw` protocol, supporting eager mode padded to `max_seq_len` and token-budget packing.

## Known Hardware Validation Boundaries (2SN Measurements, 2026-09-24)

- **`activation_checkpoint.swap_inputs=true` is incompatible with chunk-loss forward replacement**: During activation recomputation in the first
  backward, measurements produced `RuntimeError: SwapGroup 'swap_group_0' wait_offload() called
  before launch_offload()`: the forward hook in `core/activation_memory/swap.py` conflicts with the family's modified
  forward timing. The shipped recipe defaults to `swap_inputs: false` and is unaffected. To save memory at smaller edp settings such as 2SN,
  adjust sequence length, GBS, or other controls instead.
- **2SN (edp=2) has a hard memory limit and requires optimizer-state swapping**: With `ep=128`, per-device
  parameters, gradients, and optimizer states at 2SN are approximately twice those at 4SN (edp=4). Previous measurements for this family and shape required **56.59 GB without swap**,
  close to the 61.3 GB limit, with step times varying from 24.9–41.0 s and occasional OOM. **Optimizer-state swapping reduced this to
  41.49 GB / 21.06–22.84 s**. It is therefore required at this scale. This change does not include that wiring: the core primitive
  `core/optimizer/swap_optimizer.py` is already in the baseline, but Trainer-side YAML configuration is missing. Measurements on
  2SN / ep128 / seq8192 / GBS4096 therefore hit OOM during backward at approximately 55.4 GiB. Neither `fsdp_config.enable_offload`
  nor `activation_checkpoint.swap_inputs`, which is incompatible with chunk-loss forward, can replace it.
  **The main recipe is sized for 4SN**, where operation without swapping is the family's best measured configuration.
- **Multimodal batches contain only raw `labels`**: The framework's `ChunkedCausalLMLoss` requires preshifted `shift_labels`,
  so `loss_fn` must use the family subclass `adapter/loss.py::KimiChunkedCausalLMLoss`, as this recipe does.

## Dependencies and Asset Prerequisites

- **Transformers supplies the native model architecture; this family ships no modeling code**. Construction reads checkpoint `config.json` through
  `AutoConfig(trust_remote_code=False)` and resolves classes from the **configuration class**:
  the top-level class is Transformers' `Kimi_K25ForConditionalGeneration`, with nested `DeepseekV3ForCausalLM`
  (`text_config.architectures` is declared by the checkpoint and is not rewritten here).
  Transformers must provide native `kimi_k25` / `deepseek_v3`; the validated version is **5.13.0**. If unavailable, the builder
  reports the required version instead of falling back to remote code in the checkpoint directory.
- Although checkpoint directories include remote code such as `modeling_kimi_k25.py`, `modeling_deepseek.py`, and
  `kimi_k25_processor.py`, supplied by the official repository for older Transformers versions and referenced by `config.json`'s
  `auto_map`, this family **deliberately does not use it**. It differs from the native implementation in class names and processor protocols,
  requires executing Python from the model directory, and would require redoing the entire integration.
- **Weights must be BF16**: The official HF release uses 4-bit compressed-tensors quantization
  (`text_config.quantization_config`), while the builder clears that field to match dequantized BF16 weights.
  Combining quantized weights directly with BF16 configuration causes shape mismatches. Dequantize first—the repository has validated Ascend msModelSlim w4a8 →
  BF16 per-tensor dequantization—or use a quantized loading path.
- Weight loading is **strict** (`load_pretrained_weights(..., strict=True)`): key or shape mismatches raise errors.
  This also validates at runtime that the native implementation matches the official weight structure.

## Using the Recipe

`recipes/train.yaml` targets the **main scale of 4 SuperPoDs / 512 devices**, with
`cp 1 × dp_shard 512 (= edp 4 × ep 128)`, `GBS 512 = 512 × MBS 1`, and sequence length 8192.
This is the family's best measured 4SN configuration: minimum step time **18.2617 s**, padded throughput **228,873 tok/s**,
and peak allocated memory **39.19 GB**, without activation swap; `edp=4` reduces each device's share of expert parameters to one quarter.

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

## Tests

| Test | Coverage |
|---|---|
| `tests/ut/auto_models/test_kimi_k26_chunk_loss.py` | Chunk Loss binding, idempotence, guards, and numerical agreement |
| `tests/ut/auto_models/distributed/test_kimi_k26_cp_input.py` | CP input sharding contract: windows, position_ids, and masks |
| `tests/ut/data/vlm/test_dynamic_padding.py` | Dynamic padding and packing paths |
| `tests/ut/data/vlm/test_image_max_pixels.py` | Image pixel budgets and media token counts |
| `tests/ut/data/vlm/test_truncate_mode.py` | Truncation policies and sequence-length inference |
| `tests/ut/data/vlm/test_get_batch_cp.py` | CP batch sharding |
