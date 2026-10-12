# Feature Status

**English** | [简体中文](feature_status.zh.md)

This document records HyperParallel's supported and planned model and system capabilities. `[x]` means supported in the current version; `[ ]` means planned.

## Supported Models

- [x] Qwen3-30B-A3B

## HyperShard

### DTensor

- [x] DTensor basic
- [x] DTensor redistribute
- [x] manual_seed distributed random seed control
- [ ] DTensor centric communication
- [ ] Cross Mesh DTensor redistribution

### HSDP / FSDP

- [x] Parameter and optimizer sharding
- [x] Parameter, optimizer, and gradient sharding
- [x] Overlap (full overlap mode)
- [x] gradient_scaling_factor gradient scaling
- [ ] Dynamic-to-static conversion

### Shard / TP

- [x] Supported distributed operator list
- [x] Custom distributed operator registration (YAML registry + Python implementation)
- [x] Custom Shard
- [x] DFunction custom distributed Torch autograd functions
- [x] Loss Parallel for TP training

#### TP Styles

- [x] ColwiseParallel / RowwiseParallel / SequenceParallel
- [x] parallelize_module declarative TP interface
- [x] 1D
- [ ] Higher-dimensional TP: 2D/2.5D/3D

#### EP

- [x] Basic ExpertParallel / ExpertTensorParallel workflow
- [x] MoE building blocks (GroupedExperts / TokenChoiceTopKRouter / MoE)
- [x] Load balancing (expert_bias + aux_loss + AutoScaler)
- [x] MoE zero-overhead activation storage
- [x] Decoupled MoE + EP token dispatch
- [ ] Basic dropless workflow
- [ ] Compute-communication overlap
- [ ] Live expert migration / replicas for overloaded experts

#### CP

- [x] ContextParallel basic context parallelism
- [x] AsyncContextParallel asynchronous context parallelism
- [x] DSA variants (Indexer / Loss / SparseAttention)
- [x] TP DTensor local rewrap
- [ ] DeepSpeed Ulysses
- [ ] Ring Attention
- [ ] 3D sequence parallelism

- [ ] Overlap

### Distributed Random Numbers

- [x] manual_seed random seed control
- [ ] Dropout

## HyperMPMD

### Pipeline Parallelism

- [x] GPipe
- [x] 1F1B
- [x] VPP(ScheduleInterleaved1F1B)
- [x] PP+FSDP (MetaStep integration)
- [x] PipelineStage dx/dw computation
- [x] overlap_b_f compute-communication overlap (CommComputeOverlap two-thread coordinator)
- [x] batched P2P transport / overlap_p2p
- [x] PP Activation Swap
- [x] variable-layer + mixed-recompute under overlap_b_f
- [ ] ZBV
- [ ] SeqPP
- [ ] Different device counts per PP stage

### Mpipe Multimodal Parallelism

- [x] Mpipe Transpose scheduling
- [ ] Mpipe data-reordering

### Subgraph Partitioning

- [ ] Partition multimodal encoders / decoders across different devices

### Multicore Parallelism

- [ ] Multicore parallelism - O0
- [ ] Multicore parallelism - O1
- [x] Multicore optimization of MoE compute-communication overlap
- [ ] Multicore optimization of PP 1F1B compute-communication overlap

## HyperOffload

- [x] Activation Checkpoint(checkpoint / checkpoint_wrapper / CheckpointPolicy)
- [x] Activation Swap(swap / swap_wrapper / swap_tensor_wrapper / SwapManager)
- [x] Combined Activation Swap and Checkpointing configuration
- [x] Swap fusion
- [ ] SAS(Selective Activation Swap)
- [ ] SPO(Selective Parameter/Gradient/Optimizer Offload)
- [ ] Offload based on memory semantics
- [ ] Automatic Activation Swap policy generation

## Optimizer

- [x] AdamW
- [x] Muon(momentum-based optimizer)
- [x] ChainedOptimizer (chained Muon + AdamW)
- [x] get_hyper_optimizer
- [x] Sharded optimizers (FSDP / HSDP integration)
- [x] Gradient scaling factor + clip_grad enhancements

## AutoParallel

- [x] SAPP-ND: ND search (memory and performance estimation) <!-- codespell:ignore nd -->
- [x] SAPP-PPB:Pipeline Parallelism Balancing
- [ ] SAPP-Omni

## Fast Failure Recovery

- [x] DCP(Distributed Checkpoint)
  - [x] Distributed checkpoint saving/loading
  - [x] Asynchronous staging and saving
  - [x] Offline format conversion
  - [ ] Hugging Face format support
  - [ ] Conversion between sharding strategies
- [ ] Basic failure recovery workflow
- [ ] Fast process-level failure recovery
- [ ] Last-gasp diagnostics
- [ ] SDC detection

## Trainer

- [x] AutoModels TextTrainer text training
- [x] AutoModels VLMTrainer multimodal training
- [x] Callbacks(Logging / MoeMonitor)
- [x] AutoModels YAML Trainer and parallel dimension configuration

## Ecosystem Integration

- [x] LlamaFactory integration (activation recomputation and swap + HSDP)
- [x] Optional DeepSeek Ascend operator adapters

## Tools

### Numerical Monitoring

- [ ] global norm
- [ ] local norm
- [ ] local loss

### DryRun

- [ ] Memory overhead analysis
- [ ] Simulate cluster execution on one device

[Back to the project README](../README.md)
