# Kimi-K2.6 Model-Family Integration

**English** | [简体中文](integration.zh.md)

[Back to the training guide](README.md)

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
  during `bind_model`. The recipe uses [KimiChunkedCausalLMLoss](adapter/loss.py) to derive
  next-token targets from raw Omni labels or consume existing `shift_labels`.
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
