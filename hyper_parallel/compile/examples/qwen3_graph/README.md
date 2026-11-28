# Qwen3 Graph Training

These examples run Qwen3-0.6B through `HyperAutoModelForCausalLM`, `TextTrainer`,
and `GraphCompiler` on four Ascend NPUs using WikiText-2. Both recipes use
FSDP2 + TP2/SP, full activation recomputation, a global batch size of 4, and a
micro-batch size of 1.

| File | Purpose |
| --- | --- |
| [`run_qwen3_graph.sh`](run_qwen3_graph.sh) | Launch either recipe with `scripts/train_lm.py`. |
| [`train_qwen3_graph_tp2_fsdp2.yaml`](train_qwen3_graph_tp2_fsdp2.yaml) | Default fixed-shape recipe with SP, BF16, and 128-token indexed batches. |
| [`train_qwen3_graph_dynamic_tp2_fsdp2.yaml`](train_qwen3_graph_dynamic_tp2_fsdp2.yaml) | FP32 recipe with online packing and symbolic sequence lengths. |

## Prepare the resources

Install compatible PyTorch, `torch_npu`, and CANN/HCCL, then run the following
commands from the repository root.

```bash
pip install -e .
python -c 'import hyper_parallel; print(hyper_parallel.__file__)'
```

Place the [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) configuration and
matching tokenizer files in `./output/models/Qwen3-0.6B`. Both recipes use
random initialization and separate embedding and LM-head parameters so the
TP planner can derive a communication boundary for each. To change models
or tokenizers for the indexed recipe, update the vocabulary size and EOD
token ID in `dataset.model_assets.tokenizer` to match the indexed data.

Put the [WikiText-2 raw training split](https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-2-raw-v1)
in `./output/dataset/wikitext-train.jsonl`, with one `{"text": "..."}` object per
line. To convert the published training Parquet, save it as
`./output/dataset/wikitext-train.parquet` and run the following command.

```bash
python - <<'PY_WIKITEXT'
import json
from pathlib import Path

from pyarrow.parquet import ParquetFile

root = Path("output/dataset")
source = ParquetFile(root / "wikitext-train.parquet")
with (root / "wikitext-train.jsonl").open("w", encoding="utf-8") as out:
    for batch in source.iter_batches(columns=["text"]):
        for row in batch.to_pylist():
            if row["text"].strip():
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
PY_WIKITEXT
```

The online recipe reads this file directly. For the indexed recipe, tokenize
it with the same Qwen3 tokenizer.

```bash
python -m hyper_parallel.data.tools.offline_preparation \
  --dataset-name-or-path ./output/dataset/wikitext-train.jsonl \
  --output-prefix ./output/dataset/wikitext \
  --tokenizer-name-or-path ./output/models/Qwen3-0.6B
```

This writes `wikitext_text_document.bin` and `wikitext_text_document.idx`.
Pass their shared `wikitext_text_document` prefix as the indexed data path.

## Launch

The defaults above work with `bash hyper_parallel/compile/examples/qwen3_graph/run_qwen3_graph.sh`.
To select different local resources or devices, use the launcher overrides.

```bash
MODEL_DIR=/path/to/Qwen3-0.6B \
DATA_PATH=/path/to/wikitext_text_document \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
bash hyper_parallel/compile/examples/qwen3_graph/run_qwen3_graph.sh
```

For online packing, select the dynamic recipe and supply the raw training file.

```bash
CONFIG=hyper_parallel/compile/examples/qwen3_graph/train_qwen3_graph_dynamic_tp2_fsdp2.yaml \
DATA_PATH=./output/dataset/wikitext-train.jsonl \
OUTPUT_LOG=./output/qwen3_graph/dynamic.log \
bash hyper_parallel/compile/examples/qwen3_graph/run_qwen3_graph.sh
```

`MODEL_DIR` and `DATA_PATH` override `model.pretrained_model_name_or_path` and
`dataset.data_path`. The online tokenizer inherits the model path. Relative
paths are resolved from the repository root. `MASTER_PORT` defaults to `29683`,
`NPROC_PER_NODE` to `4`, and `OUTPUT_LOG` to `./output/qwen3_graph/train.log`.
Dotted overrides at the end of the command take precedence, for example
`--training.train_iters=3`. When changing the rank count, also update the
parallel degrees and batch sizes.

## Compile configuration

| Setting under `compile` | Effect |
| --- | --- |
| `enabled` and `use_joint_graph` | Set both to `true` to capture a joint forward/backward graph with GraphCompiler. Set both to `false` for an eager run. |
| `dynamic` | Capture symbolic user input dimensions. The indexed recipe uses static shapes and the online recipe sets `true`. |
| `dynamic_arg_dims` | Select symbolic dimensions by dotted argument path, for example `model_inputs.input_ids: [1]`. A mapping takes precedence over `dynamic`. |
| `fsdp_allgather_prefetch` | Launch FSDP parameter all-gathers early and delay their waits. Defaults to `true`. |
| `fsdp_overlap_grad_reduce` | Independently advance FSDP gradient reduce-scatter and delay its waits. Defaults to `true`. |
| `max_in_flight_gb` | Cap prefetched full parameter replicas before their first use, per phase, in GiB. `0.25` is 256 MiB. A value of `0` disables early launches. |

`activation_checkpoint.mode: full` enables recomputation across all Qwen3
decoder blocks. Set `activation_checkpoint.full.recompute_layer` to a list of
layer indices or closed ranges to select specific blocks. `accelerator.tp_size`,
`accelerator.sequence_parallel`, and `fsdp_config.dp_shard_size` control the
TP/SP and FSDP topology. See [HyperParallel Graph Mode](../../README.md) for the
pass pipeline and additional options.

### Internal FSDP prefetch depths

The all-gather prefetch pass reads two Python module variables when
`compile.fsdp_allgather_prefetch` is enabled. Edit these declarations directly
in the overlap pass before launching training. [Source](../../passes/overlap/overlap_param_gather.py)

```python
fsdp_forward_prefetch_depth: int = 1
fsdp_backward_prefetch_depth: int = 1
```

These values set how many parameter-consumer groups the pass looks back when
scheduling early all-gathers in the forward and backward phases, respectively.
A depth of `1` looks back one group. The attention scheduling heuristic can
choose an earlier launch point, subject to graph dependencies and the
`compile.max_in_flight_gb` budget. Setting a depth to `0` disables early launches
for that phase while still allowing the pass to move waits and views closer to
their consumers. Use the same non-negative integers on every rank before
compiling.
