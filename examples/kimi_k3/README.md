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
