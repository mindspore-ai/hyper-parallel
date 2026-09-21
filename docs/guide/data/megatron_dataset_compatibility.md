# Megatron Indexed Dataset Compatibility

HyperParallel consumes the standard Megatron text dataset pair directly:

```text
<prefix>.idx
<prefix>.bin
```

The index header is `MMIDIDX`, version 1. The sequence lengths and byte
pointers are followed by document boundaries. The optional per-sequence mode
array is accepted for files produced by multimodal Megatron tools; ordinary
text pretraining files do not contain it.

## Validate an external dataset

Run this command before configuring training. It checks the header, dtype,
sequence/document boundaries, byte pointers, index tail, and binary payload
size:

```bash
python -m hyper_parallel.data.tools.megatron_dataset \
    --input-prefix /data/megatron/my_corpus_text_document
```

The argument may be the prefix or either complete filename (`.idx` or
`.bin`). No NPU is needed for this check.

## Adapt the prefix

When a job needs a separate local data directory, validate and copy both files
with one command:

```bash
python -m hyper_parallel.data.tools.megatron_dataset \
    --input-prefix /data/megatron/my_corpus_text_document \
    --output-prefix /data/hyperparallel/indexed/my_corpus_text_document
```

The operation preserves the Megatron bytes; it does not tokenize or re-pack
the corpus. Add `--force` only when replacing an existing destination pair.

## Use it in training

Use the output prefix (or the original validated prefix) as the existing
indexed dataset `data_path`. The choice of iterator is controlled by the
existing `is_dataset_from_mr` option:

```yaml
dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_indexed_text_dataset
  data_path: /data/megatron/my_corpus_text_document
  data_config:
    seq_length: 4096
    split: "98, 1, 1"
    mock_data: false
    is_dataset_from_mr: false
    simple_blend: "no"
```

`is_dataset_from_mr: false` selects `GPTDataset`, which treats each indexed
document as variable length and builds document/sample/shuffle indices.
Megatron pre-packed records of exactly `seq_length + 1` tokens should use
`is_dataset_from_mr: true`, which selects `GPTFromMRDataset` and reads each
record directly. The `.idx` and `.bin` files must stay together and retain the
same prefix.

The compatibility reader is also available from Python:

```python
from hyper_parallel.data.tools.megatron_dataset import load_megatron_dataset

dataset = load_megatron_dataset("/data/megatron/my_corpus_text_document")
tokens = dataset[0]
```

The adapter is intentionally CPU/file-system functionality. NPU is required
only when the resulting indexed dataset is exercised by distributed training.
