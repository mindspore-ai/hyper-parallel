# HyperParallel v1.0.0 Release Notes

HyperParallel is a distributed parallel acceleration library for Ascend SuperNodes, decoupling model code from system optimization and providing unified distributed training capabilities ranging from cluster-level MPMD to in-chip multicore parallelism. v1.0.0 is the first official release, with the following core features:

## Core Features

- **DTensor** `[STABLE]`: Provides a unified stateless programming model based on distributed tensor abstraction, with automatic layout inference and cross-device redistribute, enabling transparent local/remote tensor operations.

- **FSDP** `[STABLE]`: Fully Sharded Data Parallel, sharding parameters, gradients, and optimizer states across devices with reshard_after_forward, communication fusion, and overlap modes, significantly reducing per-device memory usage.

- **Tensor Parallel (TP)** `[DEMO]`: Declarative tensor parallelism with ColwiseParallel, RowwiseParallel, SequenceParallel strategies, Loss Parallel support, and PyTorch distributed tensor API compatibility.

- **Context Parallel (CP)** `[STABLE]`: Sequence-dimension partitioning for long-context training (128K+), offering synchronous CP, asynchronous CP, and DSA (Dense Sparse Attention) variants with asynchronous compute-communication overlap.

- **Pipeline Parallel (PP)** `[STABLE]`: Supports GPipe, 1F1B, and VPP scheduling strategies, PP+FSDP integration, P2P prefetch overlap, Mpipe multimodal transpose, and forward-backward overlap for EP all-to-all communication masking.

- **Expert Parallel (EP)** `[DEMO]`: Expert parallelism for MoE models with EP+TP 2D parallelism, building blocks such as GroupedExperts and TokenChoiceTopKRouter, and support for load balancing and auxiliary loss.

- **Activation Checkpoint** `[STABLE]`: Selective recomputation via checkpoint_wrapper, trading compute for memory with policy-based layer selection, working in coordination with the swap mechanism.

- **Swap (Activation Offload)** `[STABLE]`: Asynchronously offloads activations to CPU via swap_wrapper and prefetches them back to NPU during backward pass, with SwapManager for layer-level offload/prefetch coordination.

- **Auto Parallel Search (SAPP)** `[DEMO]`: SAPP-ND provides multi-dimensional parallelism strategy search (DP/TP/PP/EP) with memory estimation; SAPP-PPB performs pipeline stage load balancing with joint recomputation tuning.

- **Mpipe Multimodal Parallel** `[DEMO]`: Add multimodal transpose scheduling for Mpipe VLM.

- **Distributed Checkpoint (DCP)** `[STABLE]`: Per-rank sharded model state saving with cross-strategy reshard loading, async staging, and offline format conversion, eliminating single-device memory bottlenecks.

- **Distributed Optimizer** `[STABLE]`: Provides AdamW and Muon optimizers with ChainedOptimizer for mixed parameter group training, gradient scaling, and FSDP-integrated sharded optimizer states.

- **In-Chip Multicore Parallelism** `[DEMO]`: Two-level on-chip MPMD parallelism with O0 (Host CPU scheduling) and O1 (AICore scheduling), combining multicore dispatch with one-sided communication to improve communication masking and MAC utilization in MoE scenarios.

- **DFunction** `[STABLE]`: Custom distributed autograd function interface with automatic DTensor dispatch, layout inference, and output wrapping, enabling users to extend custom distributed operators.

- **Deferred Weight Initialization** `[STABLE]`: Supports deferred model weight initialization by constructing the model structure on a meta device first and materializing parameters on demand, reducing peak memory usage during large model initialization.

## Platform Support

- Supports PyTorch 2.6 / 2.7 / 2.9 and MindSpore backends.
- Supports pip installation and source build with configurable native extensions (multicore, symmetric memory, custom ops).
