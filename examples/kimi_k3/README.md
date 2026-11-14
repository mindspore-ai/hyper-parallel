# Kimi K3

The native text baseline supports scratch pretraining with KDA, gated NoPE MLA,
SiTU, latent MoE and shared experts, Quantile Balancing, and block attention
residuals. It uses the standard TextTrainer and model adapter pipeline.

- [LLM pretraining](README.llm.md): local configuration and indexed text data.
- [Image and video pretraining](README.image.md): MoonViT, local media, and the VLMTrainer entry.
- [Architecture and scope](../../docs/models/kimi_k3.md).
- [Validation](../../docs/models/kimi_k3_validation.md).

The original launchers retain conservative CP1/EP1 defaults. Explicit parallel
recipes are available for [LLM](train_llm_parallel_pretrain.yaml) and
[image/video](train_media_parallel_pretrain.yaml) training with TP=PP=1.
They compose FSDP, decoder CP, optional EP and encoder parallelism through model
adapters. Review local configuration, kernel dependencies, world size and the
validation scope before selecting a recipe; available options are not an
acceptance result for every combination.

## Optional periodic evaluation

Keep the dataset and topology configuration of the chosen recipe, provide an
independent validation split, and enable a bounded evaluation window:

```yaml
training:
  eval_iters: 1
  eval_steps: 100
```

For the online image/video entry, `dataset.data_path` can name explicit splits:

```yaml
dataset:
  data_path:
    train: /path/to/train.jsonl
    valid: /path/to/valid.jsonl
```

For indexed text, set a nonzero validation share in `dataset.data_config.split`
(for example `99, 1, 0`); the supplied training-only recipe uses `1, 0, 0`. `eval_iters`
counts global batches, including all accumulation microbatches. The validation
loader must provide the entire window on every participating data rank. Evaluation
requires TP=PP=1 and no background host prefetch. It restores training RNG/modes
and revisits the saved validation cursor, without updating QB or the optimizer.
See the [validation contracts](../../docs/models/kimi_k3_validation.md) for
continuation, worker-state and numerical qualification boundaries.
