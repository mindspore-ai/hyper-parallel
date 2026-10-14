# Dynamic token batching with graph-mode text training

This example runs the standard TextTrainer + BaseTrainer + GraphCompiler path
with TP2 + FSDP2 on four NPUs. It uses an online iterable dataset,
TokenBatchLoader and dense packed attention. The model directory provides the
model configuration and tokenizer; the text generator needs no downloaded dataset.

## Run ordinary training

From the repository root in a torch + torch_npu environment:

```bash
python -m hyper_parallel.compile.examples.automodel_text_graph.prepare_dynamic_text

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 bash \
  hyper_parallel/compile/examples/automodel_text_graph/run.sh \
  hyper_parallel/compile/examples/automodel_text_graph/train_online_lm_graph_dynamic_tp2_fsdp2.yaml \
  --model.pretrained_model_name_or_path=/path/to/Qwen3-0.6B
```

The YAML enables `compile.dynamic: true`; manual dimension annotations are
optional. Packing combines variable numbers of records into `[1, packed_length]`,
so the sequence dimension changes while the physical batch dimension remains one.
The iterable dataset avoids limiting raw records before token packing.
Replace `dataset.data_path` to use another plaintext source.

This functional smoke run uses random initialization (`model.load_base_model: false`)
and FP32 model parameters. Graph FSDP shards parameters on first capture;
separate optimizer master copies constructed earlier can retain the full shape.
That optimizer combination is not exercised by this example.

`run.sh` defaults to `scripts/train_lm.py`. It accepts a YAML path followed by
existing CLI overrides. Set `PYTHON=/path/to/venv/bin/python` to select the
interpreter, and use `MASTER_PORT`, `NPROC_PER_NODE` and `OUTPUT_DIR` for launcher
settings. The default YAML remains the indexed/static example. The output is a
normal training log; this entry does not export a separate per-rank audit JSON.

## Optional size specialization

Add these overrides to the dynamic training command:

```bash
  '--compile.compile_sizes=[116,120,127]' \
  --compile.compile_size_input=model_inputs.input_ids \
  --compile.compile_size_dim=1
```

The first execution uses the general graph. Later configured sizes generate an
FX variant once per complete input metadata signature and reuse it. Other sizes,
or new signatures at the cache limit, continue using the general graph.
`compile.max_specializations` defaults to 8. Omit `compile_sizes` to disable variants.
A different tokenizer or dataset can change which packed lengths recur.

Generation binds shape expressions in the post-pass FX graph. It does not
recapture the model, shard parameters again, execute collectives during generation
or invoke an Inductor/NPU kernel compiler. Tensor values, token counts and model
weights remain runtime inputs. This is a functionality example, not a throughput
or convergence benchmark.

## Regression tests and implementation notes

Small-model shape/gradient and dispatch checks live in the existing CPU UT:

```bash
python -m pytest tests/ut/compile/test_dynamic_shapes.py \
  tests/ut/compile/test_size_specialization.py -q
```

These tests check eager loss/gradient parity, graph reuse, guards and multiple hot
sizes. CPU UT and mocked communication do not substitute for real NPU training.
Dynamic PP and arbitrary data-dependent shapes remain outside the supported path.
