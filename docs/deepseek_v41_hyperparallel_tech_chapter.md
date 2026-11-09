# III. Key HyperParallel Implementations: Integrating Model Structure into Distributed Training

**English** | [简体中文](deepseek_v41_hyperparallel_tech_chapter.zh.md)

For DeepSeek-V4.1, HyperParallel's adaptation goes beyond reproducing the model architecture: it brings cross-layer state, sparse parameters, and multimodal computation into a composable distributed training workflow. Model adapters connect parameter sharding, TP/CP/EP, recomputation, external state, vision encoder parallelism, and loss computation to a unified Trainer, while adapters and configuration compose the model-specific logic.

## 1. CSA2: Managing Cross-Layer Shared State and Parallel Communication

CSA2's Full, Reindex, and Reuse paths share compressed KV, Indexer Keys, and Top-K state across layers. HyperParallel manages this state by attention role. Under context parallelism, each rank retains local Queries and obtains the KV and Indexer Keys needed for computation through asynchronous differentiable AllGather. Under tensor parallelism, scores from sharded Indexer Heads are aggregated before global Top-K selection.

Activation recomputation follows the same state boundaries. Replay reuses the discrete indices selected during forward and avoids writing shared state again, keeping CSA2 recomputation consistent with the cross-layer state lifecycle.

## 2. Engram: CPU Residency, EP Sharding, and Sparse Optimization

Engram tables are partitioned into row ranges by expert parallel rank, with FP32 parameters optionally resident on CPU. Forward performs lookups on demand; backward records gradients only for accessed rows. Before the training step ends, duplicate rows are merged and sparse gradients synchronized, then SparseAdam updates the table. Checkpoints save and restore each rank's table data and sparse optimizer state through separate sidecars.

This path reduces the device-resident memory required by very large conditional memory tables. The cropped-model experiment in this article validates the training path; it does not mean full-scale Engram parameters have been loaded.

## 3. mHC and MoE: Integrating Ascend Fused Execution Paths

HyperParallel maps DeepSeek-V4.1's mHC modules to HyperMegaMhc for fused computation at inter-layer boundaries. It also provides MegaGate router replacement and parallel execution paths for routed and shared experts. The adaptation layer connects model parameter mappings with the framework's parallel strategies, avoiding a separate model-specific training entry point.

## 4. Reusing Fused Operators and Integrating Them at Model Level

The Ascend execution path for DeepSeek-V4.1 also reuses some fused operator implementations from [torchtitan-npu](https://github.com/torchtitan-npu/torchtitan-npu). HyperParallel's distinct contribution is integrating these operators into its functional components and model adaptation layer: handling model tensor layouts and sequence-length metadata, connecting routing, sparse indices, and output rearrangement around operators, and incorporating them into the corresponding parallel computation and automatic differentiation paths. Models can therefore reuse existing high-performance NPU kernels while HyperParallel adapters and the Trainer still organize training.

Integration paths visible in the repository include DSA Lightning Indexer and sparse Flash Attention, parameterized sink attention, and MoE grouped matmul, SwiGLU, and token permutation/restoration. The wrappers adapt operator layouts such as TND, retain softmax statistics needed for attention backward, connect grouped GEMM weight gradients with gradient accumulation, and place expert computation in the EP path. The article should describe this contribution as “fused operator reuse plus HyperParallel model and distributed training integration.” Explain the underlying operator sources separately from framework-side adaptation; do not describe reused kernels as original HyperParallel operators.

## 5. Multimodality and Loss: Partitioning Work by Computational Structure

The multimodal path provides parallel adaptation for the DeepSeek-V4.1 vision encoder and preserves the mapping from image metadata to vision work units. Training loss supports sequence-chunked cross-entropy, avoiding materialization of full-vocabulary logits for the entire sequence at once and controlling intermediate tensor overhead for large-vocabulary training.

Distinguish implemented support from the measurements in this article: throughput and memory results still come from randomly initialized cropped models and short training runs. Changes in PRs not merged into the main branch are excluded from those results.
