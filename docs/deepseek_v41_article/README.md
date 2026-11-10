# HyperParallel Supports DeepSeek-V4.1-Flash Training on Ascend: Bringing Advanced Model Architectures to Ascend Training

**English** | [简体中文](README.zh.md)

## I. Introduction: DeepSeek-V4.1-Flash Brings New Training Challenges

As large models evolve toward long context, multimodality, and agent applications, capability improvements depend on more than parameter growth. More efficient training and inference within limited compute, storage, and communication budgets have become important directions for innovation.

In September 2026, DeepSeek released **DeepSeek-V4.1-Flash**. This new multimodal MoE model has **552B backbone parameters** and supports contexts up to **1M tokens**. Architectural and optimization innovations improve both model capability and computational efficiency.

Compared with its predecessor, DeepSeek-V4.1-Flash introduces an asymmetric Causal Encoder-Decoder (CED) architecture, Compressed Sparse Attention 2 (CSA2), Engram conditional memory, and Single-Pass mHC, together with Head-wise Muon and Sinkhorn Balancing optimization methods, exploring more efficient large-scale training.
These innovations improve efficiency while posing new framework requirements: sharing parameters and computational state across layers, managing enormous sparse parameter sets, combining new optimizers with distributed training, and maintaining efficient computation and communication across multimodal, MoE, and long-sequence workloads.

**HyperParallel is a distributed parallel acceleration framework for Ascend SuperPoDs.** It decouples model code from system optimization and provides FSDP, TP, CP, EP, and PP, along with mixed precision, recomputation, heterogeneous memory management, and optimizers.

To meet DeepSeek-V4.1-Flash's architectural requirements, HyperParallel adapts model training using composable distributed parallelism and modular system design, exploring efficient training of advanced models on Ascend.

## II. Supporting Model Innovations with Efficient Training on Ascend SuperPoDs

DeepSeek-V4.1-Flash requires more than additional model layers: it needs coordinated cross-layer state transfer, parameter sharding, activation recomputation, sparse access, optimizer updates, and device operators. Building on TP, CP, EP, and FSDP, HyperParallel adapts six core technologies and combines reusable high-performance Ascend operators to connect model architecture with distributed training.

### 1. Causal Encoder-Decoder: Asymmetric Architecture for Computational Efficiency

One major innovation is the Causal Encoder-Decoder (CED) architecture, which separates input processing and output generation into different computational paths.

Conventional decoder-only models typically use the same model path for input prefill and output decoding. CED processes input tokens through a lighter encoder path, then uses the decoder to generate outputs.

This activates only about **8B parameters per token** during prefill and **16B parameters** during decoding, substantially reducing computation for long inputs.

![DeepSeek-V4.1-Flash architecture: causal encoder, decoder, and cross-layer KV paths](https://raw.githubusercontent.com/townwish4git/hyper-parallel-deepseek-v41-article/main/assets/article/deepseek-v41-architecture.png)
CED and shared attention give layers different computation and state dependencies. HyperParallel determines compression ratios, shared KV sources, and index sources from configuration, carries shared state through layer forwards, and preserves explicit module boundaries for parameter sharding, FSDP wrapping, and activation recomputation. Attention projections, shared KV projections, and Indexer parameters receive layout strategies matching their computational roles.

### 2. CSA2: Cross-Layer KV Sharing Changes Attention Computation

For long contexts, attention computation, state storage, and communication are major determinants of efficiency.

DeepSeek-V4.1-Flash introduces **Compressed Sparse Attention 2 (CSA2)**, sharing a global KV cache and sparse indices across layers to reduce duplicated attention state.

CSA2 has three modes:

- **Full**: Compute global KV and perform sparse indexing.
- **Reindex**: Reuse global KV and recompute Top-K sparse indices.
- **Reuse**: Reuse both global KV and Top-K indices, directly computing sparse attention.

Combined with FP4 KV caching, CSA2 reduces global KV cache usage to about **890 bytes/token**, approximately one quarter of DeepSeek-V4-Flash.

![CSA2 Full, Reindex, and Reuse modes and shared state](https://raw.githubusercontent.com/townwish4git/hyper-parallel-deepseek-v41-article/main/assets/article/csa2-modes.png)
![Hierarchical sparse indexing: the first layer creates a candidate pool; later Reindex layers select Top-512 within it](https://raw.githubusercontent.com/townwish4git/hyper-parallel-deepseek-v41-article/main/assets/article/csa2-hierarchical-indexer.png)

HyperParallel manages CSA2 shared state by distinguishing producers and consumers of compressed KV, Indexer Keys, candidate blocks, and Top-K results, preserving hierarchical candidate selection and the Indexer's auxiliary training objective. With TP, it coordinates local attention and aggregation across shards. With CP, differentiable communication enables cross-device access to shared KV and index state, separating communication initiation from waiting to support compute-communication coordination.

Cross-layer sharing also complicates recomputation: multiple later layers may depend on state produced by one earlier layer. HyperParallel implements **selective recomputation aware of Full, Reindex, and Reuse roles**, keeping forward selections consistent with backward dependencies while preventing replay from republishing shared state.
HyperParallel integrates Ascend sparse attention forward/backward through `torch.ops.custom.npu_sparse_flash_attention_enhance`, registered by `omni_training_custom_ops`, and handles tensor organization and gradient wiring for training semantics such as Scalar Sink. Future fusion work will integrate **CANN Lightning Indexer V2, Sparse Flash MLA forward/backward, and fused Indexer KL-gradient operators**.

### 3. Engram: Large-Scale Conditional Memory and Efficient Parameter Management

DeepSeek-V4.1-Flash introduces **Engram conditional memory**, decoupling some knowledge storage from Transformer backbone computation through deterministic N-gram hashing and sparse table lookups.

Two Engram modules introduce approximately **196B conditional memory parameters** in total.

This supports model capability but places greater demands on parameter storage, access, and optimizer-state management for enormous embedding tables.

![Official Engram architecture: N-gram hash lookup, concatenation, and context gating](https://raw.githubusercontent.com/townwish4git/hyper-parallel-deepseek-v41-article/main/assets/article/engram-architecture.png)

HyperParallel implements N-gram hashing and lookups, handling document boundaries and image-token masks to prevent semantically invalid memory indices. Engram embeddings can be row-sharded along EP: requests route to parameter owners, results return to original token positions, and backward gradients follow ownership, without gathering the entire table on every call.
For larger sparse memory, HyperParallel also provides **CPU-resident Engram parameter tables**. Rows reside in host memory; lookups deduplicate requests and transfer only hit rows. Backward aggregates sparse gradients, CPU SparseAdam performs updates, and associated parameter and optimizer states can be saved and restored.

### 4. Head-wise Muon and Sinkhorn: Optimizers Adapted to Parameter Structure

DeepSeek-V4.1-Flash also makes targeted optimizer improvements.

**Head-wise Muon**

Conventional Muon transforms updates for an entire weight matrix. Head-wise Muon partitions attention weights such as Query and Key by attention head and applies Muon updates independently to each head.

Independent preconditioning reflects each head's gradient characteristics and better accommodates heterogeneity across heads.

**Sinkhorn Balancing**

For large matrices such as Engram embeddings, token embeddings, and prediction heads, DeepSeek-V4.1-Flash uses Momentum + Sinkhorn Balancing instead of Adam-like optimization.

Iterative row and column normalization balances update scales across dimensions. Only a momentum buffer is required, reducing optimizer-state overhead for enormous parameter sets.
![Algorithm 1: momentum updates with Sinkhorn Balancing](https://raw.githubusercontent.com/townwish4git/hyper-parallel-deepseek-v41-article/main/assets/article/sinkhorn-algorithm.png)

### 5. Single-Pass mHC: Efficient Residual Connections and Simpler Computation

DeepSeek-V4.1-Flash introduces **Single-Pass mHC** on top of Manifold-Constrained Hyper-Connections (mHC).

It reorganizes updates across residual streams, combining previously separate residual updates, input mixing, and coefficient prediction into a more efficient path.

Together with fused Mega-mHC operators, this can substantially reduce intermediate activation reads and writes during inference.

At the model level, HyperParallel adapts residual-state transfer across sublayers and deferred mixing, completing a preceding sublayer's update at the next computation boundary while handling special boundaries such as Engram and model output. At the device level, it provides a **MegaMhc training forward/backward path**: task graphs, event dependencies, and execution order are built around token tiles, organizing mixing, normalization, projection, and Sinkhorn into a continuous computation pipeline.
**Basic mathematics reuses CANN primitives such as `MhcPost`, `MhcPreSinkhorn`, and `RMSNorm`; HyperParallel builds MegaKernel scheduling, fused execution, and backward wiring on top.** Custom operators enter the PyTorch training graph through torch-npu's C++ extension and ACLNN bridging mechanisms; torch-npu built-in interfaces do not independently implement the entire MegaMhc.

Combining mature device operators with framework-side pipeline orchestration provides a path to reducing fragmented operator execution overhead.

### 6. Native Multimodal MoE: Composing Parallel Strategies to Use Cluster Capacity

DeepSeek-V4.1-Flash incorporates multimodal data from pretraining, using a DeepSeek-ViT encoder for images and a vision projection module to combine them with the language backbone.

It also retains a fine-grained MoE architecture, maintaining separate expert load-balancing policies for image and text tokens.

This combination requires vision computation, language computation, expert communication, and long-sequence attention to cooperate within one training workflow.

HyperParallel adapts DeepSeek-ViT Patch Embedding, Attention, 2D RoPE, and vision projection, inserts vision features at corresponding text positions, and preserves the boundary between image tokens and Engram text memory. **Encoder Parallel** shards computation along image patch sequences, connecting to the language backbone through Ulysses layout conversion and feature return, with independent parameter-sharding boundaries for vision modules.
For MoE, HyperParallel reuses EP for expert sharding, token dispatch/combine, and Grouped GEMM, while preserving V4.1-specific **FP32 Clipped SwiGLU** semantics rather than changing expert behavior by substituting ordinary SwiGLU.

**Expert matrix computation directly reuses existing torch-npu and CANN capabilities.** HyperParallel calls `torch_npu.npu_grouped_matmul` for Grouped GEMM and its backward computation; torch-npu/op-plugin bridges the interface to CANN GroupedMatmul, which implements device computation. Clipped SwiGLU in the current Grouped GEMM path still uses HyperParallel's FP32 logic, so the entire expert MLP must not be described as one fused torch-npu operator.
For further fusion and training-objective optimization, HyperParallel has integrated some MegaGate capabilities and continues improving MegaMoe training, dynamic expert workload management, and separate image/text load-balancing updates. **Some features remain in open PRs; activation depends on the actual training configuration.**

**As models innovate, parallel frameworks must evolve with them.** By decoupling model implementations from system optimization, HyperParallel aims to let new architectures reuse mature parallel capabilities and reduce adaptation and training costs.

## III. Key HyperParallel Implementations: Integrating Model Structure into Distributed Training

For DeepSeek-V4.1, HyperParallel's adaptation goes beyond reproducing the model architecture: it brings cross-layer state, sparse parameters, and multimodal computation into a composable distributed training workflow. Model adapters connect parameter sharding, TP/CP/EP, recomputation, external state, vision encoder parallelism, and loss computation to a unified Trainer, while adapters and configuration compose the model-specific logic.

### 1. CSA2: Managing Cross-Layer Shared State and Parallel Communication

CSA2's Full, Reindex, and Reuse paths share compressed KV, Indexer Keys, and Top-K state across layers. HyperParallel manages this state by attention role. Under context parallelism, each rank retains local Queries and obtains the KV and Indexer Keys needed for computation through asynchronous differentiable AllGather. Under tensor parallelism, scores from sharded Indexer Heads are aggregated before global Top-K selection.

Activation recomputation follows the same state boundaries. Replay reuses the discrete indices selected during forward and avoids writing shared state again, keeping CSA2 recomputation consistent with the cross-layer state lifecycle.

### 2. Engram: CPU Residency, EP Sharding, and Sparse Optimization

Engram tables are partitioned into row ranges by expert parallel rank, with FP32 parameters optionally resident on CPU. Forward performs lookups on demand; backward records gradients only for accessed rows. Before the training step ends, duplicate rows are merged and sparse gradients synchronized, then SparseAdam updates the table. Checkpoints save and restore each rank's table data and sparse optimizer state through separate sidecars.

This path reduces the device-resident memory required by very large conditional memory tables. The cropped-model experiment in this article validates the training path; it does not mean full-scale Engram parameters have been loaded.

### 3. mHC and MoE: Integrating Ascend Fused Execution Paths

HyperParallel maps DeepSeek-V4.1's mHC modules to HyperMegaMhc for fused computation at inter-layer boundaries. It also provides MegaGate router replacement and parallel execution paths for routed and shared experts. The adaptation layer connects model parameter mappings with the framework's parallel strategies, avoiding a separate model-specific training entry point.

### 4. Reusing Fused Operators and Integrating Them at Model Level

The Ascend execution path for DeepSeek-V4.1 also reuses some fused operator implementations from [torchtitan-npu](https://github.com/torchtitan-npu/torchtitan-npu). HyperParallel's distinct contribution is integrating these operators into its functional components and model adaptation layer: handling model tensor layouts and sequence-length metadata, connecting routing, sparse indices, and output rearrangement around operators, and incorporating them into the corresponding parallel computation and automatic differentiation paths. Models can therefore reuse existing high-performance NPU kernels while HyperParallel adapters and the Trainer still organize training.

Integration paths visible in the repository include DSA Lightning Indexer and sparse Flash Attention, parameterized sink attention, and MoE grouped matmul, SwiGLU, and token permutation/restoration. The wrappers adapt operator layouts such as TND, retain softmax statistics needed for attention backward, connect grouped GEMM weight gradients with gradient accumulation, and place expert computation in the EP path. The article should describe this contribution as “fused operator reuse plus HyperParallel model and distributed training integration.” Explain the underlying operator sources separately from framework-side adaptation; do not describe reused kernels as original HyperParallel operators.

### 5. Multimodality and Loss: Partitioning Work by Computational Structure

The multimodal path provides parallel adaptation for the DeepSeek-V4.1 vision encoder and preserves the mapping from image metadata to vision work units. Training loss supports sequence-chunked cross-entropy, avoiding materialization of full-vocabulary logits for the entire sequence at once and controlling intermediate tensor overhead for large-vocabulary training.

Distinguish implemented support from the measurements in this article: throughput and memory results still come from randomly initialized cropped models and short training runs. Changes in PRs not merged into the main branch are excluded from those results.

## IV. Training Practice: From Model Construction to Distributed Training

After implementing the architecture and core training components, HyperParallel provides end-to-end DeepSeek-V4.1-Flash examples covering **text-only training, multimodal training, and TP, CP, EP, FSDP, and other distributed capabilities**.

A unified Trainer interface, declarative YAML configuration, and modular parallel strategies handle model construction, component replacement, data processing, distributed initialization, and training without requiring manual orchestration of complex parallel logic in the entry point.

The following workflow uses the validated **single-node, eight-device Ascend 910B3 NPU environment, with 64 GiB HBM per device**. Results below use the same eight-device environment, and demo configurations match the measured configurations.

### 1. Model and Environment Preparation

To quickly validate training for the new architecture, HyperParallel provides the cropped-model builder `build_cropped_deepseek_v41()`.

It reads the official configuration, scales selected parameter dimensions while preserving key structures and layer roles, and randomly initializes the model through `HyperAutoModelForCausalLM.from_config()`.

The examples support text-only and multimodal scenarios:

| Configuration | Text-only training | Multimodal training |
| -------------- | ---------------- | ---------------- |
| Language model layers | All 40 retained | All 40 retained |
| Vision model layers | — | All 32 vision blocks retained |
| Text parameter dimension scaling | 1/8 | 1/4 |
| Vision parameter dimension scaling | — | 1/4 |
| Routed Experts | 16               | 16               |
| Maximum sequence length | 4096 | 4096 |
| Measured devices | 8 × Ascend 910B3 | 8 × Ascend 910B3 |

Scaling primarily affects model width, head count, low-rank dimensions, and related specifications; total parameter count does not necessarily shrink by the same ratio. Backbone depth and specified layer roles remain intact, allowing CSA2, Engram, mHC, and MoE training paths to be validated with fewer resources.

**This example constructs and randomly initializes a cropped model from the official configuration; it does not load pretrained weights for the full 552B-parameter backbone.** The experiment validates architecture, distributed training paths, and resource usage. Full-scale training still requires additional resource planning and numerical and stability validation.

First obtain HyperParallel and configure the Ascend training environment:

```bash
git clone https://github.com/mindspore-ai/hyper-parallel.git
cd hyper-parallel

source /path/to/Ascend/cann/set_env.sh
python -m pip install -e .
```

Then prepare the official DeepSeek-V4.1-Flash configuration and tokenizer, and set the local model path:

```bash
export DSV41_MODEL_DIR=/path/to/DeepSeek-V4.1-Flash
export PYTHON_BIN=python
export NPROC_PER_NODE=8
```

The measured environment uses Ascend 910B3, CANN 9.1.0, PyTorch 2.9.0, torch-npu 2.9.0.post6, and Transformers 5.13.0, with a matching `omni_training_custom_ops` installation for enhanced sparse attention and related operators.

### 2. Text-Only Training: TP, CP, and Other Parallel Configurations

HyperParallel provides a unified text-training entry point for DeepSeek-V4.1-Flash, with run modes to switch quickly between TP, CP, and other strategies.

The experiment uses eight Ascend 910B3 devices, global batch size 8, and micro-batch size 1, adjusting EP and FSDP sharding for each strategy.

Start with TP1 training:

```bash
NPROC_PER_NODE=8 bash \
  examples/training_demo/deepseek_v41/run_deepseek_v41_online.sh \
  "$DSV41_MODEL_DIR" tp1 \
  --training.train_iters=6 \
  --training.global_batch_size=8 \
  --accelerator.ep_size=8 \
  --fsdp_config.dp_shard_size=8 \
  --checkpoint.restore_from=null \
  --checkpoint.save_ckpt=false \
  --dataset.data_path=/path/to/online_4k_aligned.jsonl
```

The script automatically checks model assets, prepares Engram metadata, starts distributed processes, constructs the model, and replaces components. `TextTrainer` then runs forward, backward, gradient synchronization, and optimizer updates.

After TP1 validation, switch to TP2 or CP2:

```bash
# TP2 + Sequence Parallel
NPROC_PER_NODE=8 bash \
  examples/training_demo/deepseek_v41/run_deepseek_v41_online.sh \
  "$DSV41_MODEL_DIR" tp2 \
  --training.train_iters=6 \
  --training.global_batch_size=8 \
  --accelerator.ep_size=8 \
  --fsdp_config.dp_shard_size=4 \
  --checkpoint.restore_from=null \
  --checkpoint.save_ckpt=false \
  --dataset.data_path=/path/to/online_4k_aligned.jsonl
# CP2
NPROC_PER_NODE=8 bash \
  examples/training_demo/deepseek_v41/run_deepseek_v41_online.sh \
  "$DSV41_MODEL_DIR" cp2 \
  --training.train_iters=6 \
  --training.global_batch_size=8 \
  --accelerator.ep_size=8 \
  --fsdp_config.dp_shard_size=8 \
  --checkpoint.restore_from=null \
  --checkpoint.save_ckpt=false \
  --dataset.data_path=/path/to/online_4k_aligned.jsonl
```

The three measured text-only configurations are:

| Mode | TP | CP | EP | FSDP shards | Global Batch | Gradient accumulation | Sequence Parallel |
| --- | -: | -: | -: | ------: | -----------: | ---: | ----------------- |
| TP1 | 1 | 1 | 8 | 8 | 8 | 1 | Disabled |
| TP2 | 2 | 1 | 8 | 4 | 8 | 2 | Enabled |
| CP2 | 1 | 2 | 8 | 8 | 8 | 2 | Disabled |

All modes use PP=1 and micro-batch size 1. TP2 shards computation and intermediate state through tensor and sequence parallelism; CP2 partitions the sequence dimension, using the shared-attention KV AllGather path in this baseline.
Device counts and parallel settings must change together. Modifying only `NPROC_PER_NODE` does not automatically update EP, FSDP shard count, or global batch size; adjust the corresponding topology parameters when changing scale.

### 3. Multimodal Training: Unified Vision and Language Training

HyperParallel also provides an image-text multimodal training example for DeepSeek-V4.1-Flash.

A unified `VLMTrainer` manages the vision encoder, language model, image-text preprocessing, and parameter optimization.

Training data uses JSONL in OpenAI Messages format, for example:

```json
{"messages":[{"role":"user","content":[{"type":"text","text":"Describe the object in the image."},{"type":"image_url","image_url":{"url":"images/example.png"}}]},{"role":"assistant","content":"There is a cat in the image."}]}
```

After preparing images and JSONL data, launch eight-device multimodal training:

```bash
RUN_NAME=dsv41_vlm_10steps \
NPROC_PER_NODE=8 \
HCCL_NPU_SOCKET_PORT_RANGE=auto \
bash examples/training_demo/deepseek_v41/run_deepseek_v41_vlm_online.sh \
  "$DSV41_MODEL_DIR" /path/to/v41-data/train.jsonl \
  --training.train_iters=10 \
  --training.global_batch_size=8 \
  --accelerator.ep_size=8 \
  --fsdp_config.dp_shard_size=8 \
  --checkpoint.restore_from=null \
  --checkpoint.checkpoint_dir=./output/deepseek_v41/vlm_checkpoints
```

The multimodal experiment uses **TP=1, CP=1, EP=8, eight FSDP shards, and global batch size 8**, completing ten optimizer updates.

During training, HyperParallel uses YAML `plan_overrides` to replace and parallel-wrap mHC, Engram, shared compressed attention, and other components, binding CSA2 parallel strategies, Engram EP lookups, and MoE expert computation.

Image-text processing, model construction, distributed initialization, forward, backward, optimizer updates, and checkpoint saving all run through the unified Trainer, without additional manual parallel orchestration in the training entry point.

### 4. Results: Training Metrics and Model Correctness Validation

On **one node with eight Ascend 910B3 NPUs, each with 64 GiB HBM**, HyperParallel validated text-only and multimodal training for DeepSeek-V4.1-Flash. Randomly initialized cropped models retain key modules and network depth, using synthetic data to validate the end-to-end workflow.

**(1) Text-only training: validating multiple parallel strategies**

TP1, TP2 + Sequence Parallel, and CP2 each completed six training steps with finite losses and gradient norms.

| Metric | TP1 | TP2 + SP | CP2 |
| ------------ | -------: | -------: | -------: |
| Mean step time (s) | 19.57 | 38.71 | 23.30 |
| Throughput (tokens/s) | 1,674.77 | 846.56 | 1,406.62 |
| Peak allocated memory (GiB) | 39.08 | 24.42 | 20.65 |

TP1 achieved the highest throughput. Compared with TP1, CP2 reduced peak allocated memory by approximately **47%**, with about 19% higher mean step time, demonstrating the tradeoff between training performance and memory usage.

![Performance comparison across parallel strategies](https://raw.githubusercontent.com/townwish4git/hyper-parallel-deepseek-v41-article/main/assets/article/deepseek-v41-parallel-performance.png)

**(2) Multimodal training: end-to-end workflow validation**

Multimodal training completed ten optimizer updates. Loss decreased from **12.1633 to 10.5255**, with finite loss and gradient norms throughout.

| Metric | Measured result |
| ------ | --------: |
| Completed steps | 10/10 |
| Mean step time | 5.76 s |
| Peak allocated memory | 9.25 GiB |
| Peak reserved memory | 11.03 GiB |

![Multimodal training loss curve](https://raw.githubusercontent.com/townwish4git/hyper-parallel-deepseek-v41-article/main/assets/article/deepseek-v41-vlm-loss.png)

![Per-step multimodal training metrics](https://raw.githubusercontent.com/townwish4git/hyper-parallel-deepseek-v41-article/main/assets/article/deepseek-v41-training-log.png)
The experiment also saved a multimodal training checkpoint successfully, and relevant model unit tests passed, further validating the executability of the components and training workflow.

**These measurements validate HyperParallel's end-to-end support for DeepSeek-V4.1-Flash text-only, multimodal, and multidimensional parallel training.** Results use cropped models, synthetic data, and short runs. Full-scale performance, long-run convergence, numerical alignment across strategies, and checkpoint restoration require further validation.

*Note: Mean step time excludes the first step; text throughput counts valid tokens. Multimodal samples do not fill the 4K context, so step times for the two training types should not be compared directly.*

## V. Conclusion: Bringing Advanced Model Innovations into Practice

From Causal Encoder-Decoder and CSA2 to Engram, Head-wise Muon, and Sinkhorn Balancing, DeepSeek-V4.1-Flash demonstrates continued innovation in architecture and training optimization.

Growing architectural complexity also demands more flexible, composable, and extensible training frameworks.

**HyperParallel continues building unified, efficient, scalable distributed training infrastructure around Ascend SuperPoD architectures.** Decoupled model/system optimization, modular adaptation, and multidimensional parallel composition support MoE, multimodality, very long sequences, and new optimizers.

HyperParallel will continue improving complex-model adaptation, distributed optimizers, sparse parameter management, mixed precision, and compute-communication overlap to make large-scale training easier and more efficient.

**Make advanced models easier to train and heterogeneous computing resources more effective.**

Join the HyperParallel open-source community to explore more possibilities in large-model training.

**Open-source project:**
<https://github.com/mindspore-ai/hyper-parallel>

**DeepSeek-V4.1-Flash technical report:**

<https://arxiv.org/abs/2609.19969>

**Star, fork, and contribute through issues and pull requests!**
