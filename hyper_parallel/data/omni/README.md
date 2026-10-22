# Opt-in multimodal data pipeline

This directory extracts the generic multimodal data components from
[HyperParallel PR #884](https://github.com/mindspore-ai/hyper-parallel/pull/884)
(merge commit `edd4fae93147567ccf08507db2d095810b8d6938`). It adapts them to this
branch's existing dataset builder and Torch StatefulDataLoader. It does **not**
replace `data/text` or any text training YAML. The transitional `data/vlm`
interfaces have been removed; `VLMTrainer` now uses the Omni processor and batch
adapter by default. External VLM configurations referencing old `data.vlm`
targets must migrate to the example below; there are no compatibility aliases.

## Included components

| Component | Responsibility |
| --- | --- |
| `OmniDataTransform` | Eager encoding, deferred encoding after selection, and batch encoding hooks |
| `AutoProcessorTransform` | Processor-backed conversation encoding and final-assistant supervision |
| `OmniMappingDataset` | Local JSON/JSONL/Parquet indexing, relative media paths, and lifecycle delegation |
| `OmniPackingLoader` | Candidate selection, packing, and checkpointing of unread buffered samples |
| `SamplePacker` | Concatenate fields, offset media token starts, and produce `cu_seq_lens` |
| `OmniCollator` | Stack token fields and concatenate modality fields without dtype conversion |
| `OmniParallelBatch` | CP token slicing, TP metadata/tensor transport, and model runtime hooks |

The PR's common online source refactor, text transforms/loaders, and model-specific
vision/attention changes are intentionally not included. These components use
Torch tensors and the existing Torch VLMTrainer; they are not MindSpore adapters.

## Configuration

[`examples/data/omni_data.yaml`](../../../examples/data/omni_data.yaml) is a
**data-section example**, not a standalone model training recipe. Copy its
`dataset` and `dataloader` sections into a VLM configuration and supply the actual
model/processor and data paths. Use `VLMTrainer`; do not replace text YAML sections.

The example selects one complete sample per emitted batch (`token_budget: 1`),
with `micro_batch_size: 1` in the parent training configuration. The budget is a
selection budget, not a truncation length: a sample larger than the budget is
emitted alone. `AutoProcessorTransform` rejects samples exceeding `max_seq_len`
instead of cutting through a media span. Start with TP=CP=PP=1.

Sources are local `.jsonl` records, a `.json` list, or a `.parquet` file (requires
the optional `pyarrow` package). Parquet keeps row-group/row indices and caches
at most one row group per worker, without decoding images during indexing.
Choose suitably sized row groups when preparing large image datasets; each
worker's cache can hold a full row group's compressed image bytes.
`data_path` may also be an
explicit `{train: ..., valid: ..., test: ...}` mapping. Each default-transform
record contains `messages`; its final message must be an assistant response.
Relative image/video/audio paths are resolved against the source file directory.
The default transform supervises **only the final assistant response**, following
the extracted PR. This does not change our text path's multi-turn supervision.
Processor templates must preserve the token prefix produced by
`add_generation_prompt=True`; incompatible templates are rejected, not silently
masked. Existing text-specific tools/reasoning fixes are not automatically used
by this separate processor path.

### Embedded images in Parquet

The default transform also accepts records with string message contents and a
separate `images` list, for example this Python representation of a Parquet row:

```python
{
    "messages": [
        {"role": "user", "content": "Compare <image> with <image>."},
        {"role": "assistant", "content": "The first image ..."},
    ],
    "images": [
        {"bytes": png_bytes, "path": None},
        {"bytes": None, "path": "images/second.png"},
    ],
}
```

`<image>` occurrences in strings or typed text blocks consume `images` in order
across the entire conversation. Bare `{"type": "image"}` blocks also consume
one entry. Bytes are decoded to fully loaded RGB PIL images before calling the
processor; a non-null `bytes` value takes precedence over `path`. Path-only entries
(including plain path strings) are resolved relative to the source file.
The transform converts string contents to typed text/image blocks for this
external-images format, including assistant messages. Explicit image blocks
with their own source remain independent and do not consume the list.
Missing/extra images and malformed bytes raise errors rather than dropping
images or guessing their placement. Existing JSON/JSONL records with explicit
media blocks need no format changes. Set `dataset.data_path` to the Parquet file;
the transform and loader targets do not change. This support does not change the
pure-text reader or its configuration.

## Packing and model integration boundary

Multiple samples can be packed by raising `token_budget` and
`min_buffered_samples`. `cu_seq_lens` identifies original sequence boundaries;
concatenating samples alone does not isolate attention. Therefore multi-sample
packing and CP>1 require an explicit `dataloader.get_batch.runtime_input_adapter`.
The adapter can be a configured `_target_` object exposing:

```python
def build(self, *, batch, parallel_context) -> dict:
    # Model-owned positions, packed attention, and media mapping.
    ...
```

Returned fields extend model inputs; existing fields cannot be silently replaced.
The adapter sees device-resident token fields after CP slicing, full modality
fields, and global `cu_seq_lens`/media offsets. It must match the particular
model's forward contract, attention implementation, and loss/label shifting.
In particular, retaining full pixels does not by itself implement distributed
vision embedding insertion. No model-specific runtime adapter is bundled here.

CP currently accepts token fields shaped `[1, sequence]`, pads to a multiple of
`2 * cp_size`, and takes contiguous slices. Modality fields are not sequence-sliced.
Model-specific multi-axis position IDs require adapter handling. TP transport
preserves each tensor's dtype and shape. PP and encoder-DP are explicitly rejected.
Do not use dynamic packing across DP ranks without also ensuring compatible
iteration counts and token-based loss normalization in the model/trainer recipe.

## Resume and validation

Buffer state is saved with the source StatefulDataLoader state. Mapping sources
default to replay by source index (`save_by_idx: true`); full-sample buffer storage
is also available. Index replay requires unchanged data, deterministic encoding,
and the same DP world size. Unselected candidates are retained until consumed;
the dynamic buffer is drained at source exhaustion, even with `drop_last: true`.

CPU tests in `tests/ut/data/omni/test_omni_data.py` cover encoding, field packing,
buffer resume, and mocked TP/CP contracts. They do not establish multi-device
model training correctness; a real VLM forward/backward and resume test is still
required when selecting a model runtime adapter.
