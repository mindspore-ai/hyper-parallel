# HyperParallel v1.0.0 Release Notes

**English** | [简体中文](hyper_parallel_v1.0.0_release_notes.zh.md)

**Release date: 2026-06-30**

HyperParallel v1.0.0 is the project's first official release, marking the transition from rapid iteration to stable delivery. This release delivers the complete feature set of a distributed parallel acceleration library optimized for Ascend SuperPoDs, spanning cluster-level distributed parallelism and on-chip multicore parallelism, data parallelism and multidimensional hybrid parallelism, and acceleration for both training and inference.

---

## Core Features (Complete List)

### HyperShard: Decoupling Model and System Optimizations

**DTensor Distributed Tensors**

- DTensor fundamentals: a distributed tensor abstraction encapsulating a local shard, DeviceMesh, and Placements
- DTensor redistribution: on-demand redistribution across mesh dimensions, with cache optimization (`compact_str + rank_id`)
- DeviceMesh: multidimensional device-topology management; `init_device_mesh` creates meshes of arbitrary shapes
- Layout: tensor-to-mesh placement mappings for declaratively defining parallel strategies
- `manual_seed`: distributed random-seed control with PyTorch parity
- `init_parameters` / `init_empty_weights` / `init_on_device`: sharded-parameter initialization utilities

**HSDP/FSDP Data Parallelism**

- `fully_shard`: fully shard parameters, gradients, and optimizer states to reduce per-device memory usage
- `HSDPModule`: hybrid sharded data-parallel module wrapper
- `hsdp_sync_stream`: gradient-synchronization stream management
- HSDP Overlap: full overlap mode for concurrent communication and computation
- `set_gradient_scaling_factor`: configurable gradient scaling factor
- Hybrid-parallel parameter sharding: FSDP streams correctly handle mixed parallel parameters in TP+DP scenarios
- MindSpore parameter-hook migration and fixes

**Shard Distributed Operators and Tensor Parallelism**

- `shard_module`: declarative parallel-strategy interface using `sharding_plan` to configure input, output, and parameter placements
- `custom_shard`: integrate custom parallelism into the DTensor parallel workflow
- `DFunction`: base class for custom distributed autograd functions, integrated with DTensor dispatch
- `parallelize_value_and_grad`: parallelized value and gradient computation
- `SkipDTensorDispatch`: bypass DTensor dispatch in gradient hooks to operate directly on local tensors
- Distributed operator registration through a YAML registry and Python implementations
- Distributed operators including AllGatherMatmul / MatmulReduceScatter (MC2), RotaryPositionEmbedding, and more
- MindSpore replicate allreduce overlap

**Tensor Parallelism (Declarative TP Styles)**

- `ColwiseParallel`: column-wise sharding
- `RowwiseParallel`: row-wise sharding
- `SequenceParallel`: sequence parallelism
- `PrepareModuleInput` / `PrepareModuleOutput` / `PrepareModuleInputOutput`: module input/output preparation hooks
- `parallelize_module`: declarative TP application interface supporting fnmatch glob patterns
- Loss Parallel: parallel loss computation for TP training

**Expert Parallelism**

- `ExpertParallel`: standard all-to-all EP, with each rank holding `num_experts / ep_degree` local experts
- `TensorParallel` (within EP): TP-only weight sharding
- `ExpertTensorParallel`: two-dimensional EP+TP parallelism on a 2-D `[ep, tp]` mesh
- MoE building blocks: `FeedForward`, `GroupedExperts`, `TokenChoiceTopKRouter`, and `MoE`
- Load balancing: `expert_bias`, `tokens_per_expert`, and auxiliary load-balance loss (`MoEAuxLossAutoScaler`)
- Decoupled MoE+EP token dispatch: token permutation and dispatch separated from forward
- Safe state across PP+EP micro-batches: `DispatchContext` prevents instance-sharing issues

**Context Parallelism**

- `ContextParallel`: basic context parallelism
- `AsyncContextParallel`: asynchronous context parallelism
- DSA family:
  - `DSAIndexerContextParallel` / `AsyncDSAIndexerContextParallel`
  - `DSAIndexerLossContextParallel` / `AsyncDSAIndexerLossContextParallel`
  - `DSASparseAttentionContextParallel` / `AsyncDSASparseAttentionContextParallel`
- TP DTensor local rewrapping support

---

### HyperMPMD: Cluster MPMD and Multicore MPMD

**Pipeline Parallelism**

- `PipelineStage`: pipeline-stage wrapper supporting dx/dw computation
- `Schedule1F1B`: 1F1B scheduling
- `ScheduleInterleaved1F1B`: interleaved 1F1B scheduling (VPP)
- `ScheduleGPipe`: GPipe scheduling
- `MetaStep` / `MetaStepType` / `BatchDimSpec`: scheduling-unit abstractions
- PP+FSDP support: `MetaStep` integrates FSDP metasteps for parameter sharding within PP stages
- PP communication/computation overlap (`overlap_b_f`):
  - Composite B/F steps (`OVERLAP_B_F`)
  - `CommComputeOverlap` two-thread coordinator
  - `HookCoordinator` COMM-first rendezvous
  - Concurrent EP A2A and computation
  - PP P2P prefetch:`overlap_p2p=True`, batched P2P transport (duplex default under overlap_b_f)
  - forward-side P2P prefetch via `batch_isend_irecv`
- PP Activation Swap: activation swapping for pipeline parallelism
- Variable-layer + mixed-recompute under overlap_b_f
- SAPP-PPB: automatic Pipeline Parallelism Balancing module
- Batch-size validation: the MindSpore backend rejects batch sizes not divisible by `micro_batch_num`

**MoE Multicore Parallelism**

- Multicore MPMD parallelism: on-chip multicore parallelism with core-level one-sided communication using memory semantics
- Multicore optimization of MoE communication/computation overlap: fuse AllToAll-Dispatch / GMM1 / SwiGLU / GMM2 / AllToAll-Combine into one kernel, with fine-grained overlap between AIC and AIV cores
- RATR (Rank-Aware Tile Reordering): AllToAll tile reordering generated offline with zero runtime overhead

---

### HyperOffload: Separating Computation and State

**Activation Checkpoint / Swap**

- `checkpoint` / `checkpoint_wrapper`: selective activation checkpointing (SAC)
- `swap` / `swap_wrapper` / `swap_tensor_wrapper`: activation swapping (offload to CPU and prefetch during backward)
- `CheckpointPolicy`: recomputation policy configuration
- `SwapManager`: swap manager
- Coordinated configuration of activation swapping and checkpointing
- Swap fusion: fused swap-operation optimization
- Shared-function exclusion: exclude shared functions from callable overlap tracking
- LlamaFactory integration: activation recomputation and swapping support
- MindSpore support for independent recomputation scheduling

---

### Mpipe Multimodal Parallelism

- Mpipe VLM multimodal Transpose scheduling

### Optimizers

- `AdamW`: standard AdamW optimizer
- `Muon`: momentum-based Muon optimizer
- `ChainedOptimizer`: chained optimizer combining Muon and AdamW
- `get_hyper_optimizer`: optimizer factory function
- `get_hyper_lr_scheduler`: learning-rate scheduler factory function
- Sharded optimizers: sharded optimizer-state management integrated with FSDP/HSDP
- Gradient scaling factor support
- Enhanced gradient clipping: align `clip_grad_norm_` reduction with each gradient's process group

---

### DCP Distributed Checkpointing

- Distributed checkpoint saving and loading (planner, storage, and resharding)
- `async_staging`: asynchronous staging for checkpoint saving
- `offline_transform`: offline format conversion
- Hugging Face format support
- Conversion between sharding strategies

---

### AutoParallel

- **SAPP-ND**: ND search module including memory and performance estimation <!-- codespell:ignore nd -->
- **SAPP-PPB**: Pipeline Parallelism Balancing for automatically balancing pipeline-stage assignments

---

### One-sided Communication

- `Symmetric Memory`: one-sided communication with symmetric-memory semantics
- `AllGather`: one-sided AllGather operation
- `AllGatherMatmul` / `MatmulReduceScatter` (MC2): fused communication operators

---

### Process Group Management

- `init_process_group` / `destroy_process_group`
- `get_process_group_ranks` / `get_backend`
- `split_group` / `get_group_local_rank` / `mark_created_groups`

---

### Supported Models

- DeepSeekV3
- Qwen3 family:
  - Qwen3.5-0.8B-Base
  - Qwen3.5-35B-A3B-Base
  - Qwen3-VL-30B-A3B-Instruct

---

### Integrations

- LlamaFactory integration: activation recomputation, swapping, and HSDP support
- MindSpore llama3 TP+FSDP demo
- Torch llama3 PP+FSDP+CP demo
- PP+EP 1F1B correctness-verification example
- MoE Expert Parallel correctness demo

---

### Trainer Framework

- `LLMTrainer`: LLM training framework
- `VLTrainer`: multimodal vision-language training framework
- Callbacks:LoggingCallback, MoeMonitorCallback
- `parallel_dims`: parallel-dimension configuration

---

### Platform Abstraction

> **Status update: removed after v1.0.0.** The `hyper_parallel/platform/` abstraction, `get_platform()`, and the MindSpore backend
> have been removed from the main branch. The library now uses native PyTorch directly. The following describes the state at the v1.0.0 release and is retained only as historical information.

- Dual-backend support: PyTorch (GPU/NPU) and MindSpore (Ascend NPU)
- Unified abstraction through `get_platform()`
- Platform-specific implementations:
  - FSDP/HSDP: Torch and MindSpore backends
  - Pipeline Parallel: Torch and MindSpore backends
  - Activation Checkpoint/Swap: Torch and MindSpore backends
  - Process Group:Gloo + HCCL
- Wheel packaging: Python/architecture tags; framework adapters follow the corresponding framework's CXX11 ABI contract

---

## Major Changes from v0.2.0 to v1.0.0 (Incremental)

### New Features

- **Optimizer module**: chained Muon+AdamW optimizer, learning-rate schedulers, and sharded optimizers
- **Activation Checkpoint refactoring**: new `checkpoint_wrapper` / `swap_wrapper` / `swap_tensor_wrapper` interfaces
- **PP enhancements**: FSDP metastep integration, batched P2P transport, `overlap_b_f` communication/computation overlap, and activation swapping for PP
- **Context Parallel enhancements**: async CP, the DSA family (Indexer/Loss/SparseAttention), and TP DTensor local rewrapping
- **New Qwen3-family model support**: Qwen3.5/Qwen3.5-MoE/Qwen3-VL-MoE
- **MoE+EP enhancements**: decoupled token dispatch, load-balance auxiliary loss and expert-bias synchronization, and zero-overhead activation storage
- **Distributed Ops**:AllGatherMatmul / MatmulReduceScatter (MC2), RotaryPositionEmbedding, Detach, StopGradient
- **HSDP Overlap**: full overlap mode
- **Loss Parallel**: parallel loss computation for TP training
- **gradient_scaling_factor**: configurable gradient scaling factor
- **Mpipe**: heterogeneous scheduling for multimodal pipeline parallelism
- **SAPP-PPB / SAPP-ND**: automatic pipeline balancing and ND search <!-- codespell:ignore nd -->
- **LlamaFactory integration**: activation optimization and HSDP
- **Trainer framework**: LLMTrainer and VLTrainer

### Important Fixes

- **DTensor**: in-place random operators return self; fixed strided-shard `full_tensor` concatenation; parameter casts preserve DTensor identity
- **FSDP/HSDP**: fixed device-mesh construction in the FSDP meta context for `tp>1/dp=1`, noncontiguous all-reduce, mixed-parallel parameter sharding, and parameter-hook migration
- **Shard**: fixed replicated-parameter state tracking; preserve DTensor identity for in-place operators when bypassing dispatch; overlap MindSpore replicate all-reduce
- **Activation Checkpoint**: exclude shared functions from callable overlap tracking; fixed recomputation packing of sharded parameter views
- **Pipeline**: fixed the default PipelineStage device and added batch-size divisibility validation
- **Context Parallel**: fixed preservation of TP layouts in async CP
- **Optimizer**: align `clip_grad_norm_` reduction with each gradient's process group
- **Packaging**: Python/architecture wheel tags; framework adapters follow the corresponding framework's CXX11 ABI contract
- **Codecheck**: added docstrings in 66 files to resolve C0116 warnings

### Improved Test Coverage

- Activation Checkpoint UT:17.9% → 95% coverage
- Shard UT/ST: increased coverage, restructured tests, and migrated to the Gloo CPU backend (206 cases)
- Context Parallel UT: CP unit-test coverage
- PP+FSDP+TP combined numerical-accuracy tests against a single-device baseline
- Fixed Process Group worker paths and improved world-group teardown safety
- Added Trainer / trainer/utils / core/utils UT and ST

---

## Contributors

Thank you to the following contributors to v1.0.0, ordered by commit count:

| Contributor | Commits |
|--------|----------|
| xuxinglei | 26 |
| yide12 | 20 |
| Meng107 | 14 |
| lichen666 | 12 |
| DavidFFFan | 12 |
| david-he91 | 12 |
| ch-l | 10 |
| changzherui1 / changzherui | 7 |
| 宋佳琪 | 5 |
| silkage_jiajia | 5 |
| yao_yf | 4 |
| wangleiit03 | 4 |
| hedongdong | 4 |
| zhuangqinghe | 2 |
| zhangyuguo | 2 |
| yuzhenhua666 | 2 |
| yanxr123 | 2 |
| wang_hua_2025 | 2 |
| rongyue | 2 |
| liuchongming74 / liuchongming | 2 |
| Liang_Ziyi | 2 |
| lei_xxx | 2 |
| jinxiaoxian | 2 |
| guangpengz | 2 |
| Cui-yshoho / celiaccui | 2 |
| zhangbuxue | 1 |
| yuzhenhua | 1 |
| SHUYUAN6 | 1 |
| sargerasking | 1 |
| qhzhuang11111111 | 1 |
| pengshuyuan | 1 |
| niujunhao | 1 |
| liuyanwei / liu-yanwei6 | 1 |
| huilan_li / huilan li | 1 |
| duchengyi | 1 |
| buxue | 1 |
| bj-wang1 / bj-wang | 1 |
| alpha-junh | 1 |

**Total: 187 commits, 1040 files changed, 103628 insertions, 23160 deletions**

---

## Merged PRs

| PR | Description |
|----|------|
| !460 | `feat: add activation recompute & swap support for LlamaFactory integration` |
| !524 | `feat: SAPP-PPB for Pipeline Parallelism Balancing` |
| !622 | `[feature][pipeline_parallel] pp support fsdp` |
| !627 | `[feature][fully_shard] support set_gradient_scaling_factor` |
| !631 | `merge feat-sapp-nd-pr1 into master` |
| !635 | `full overlap for hsdp` |
| !636 | `merge test-gloo into master` |
| !637 | `swap_fusion` |
| !639 | `fix(shard): skip unsharded replicate params` |
| !640 | `merge feat/moe-load-balance into master` |
| !647 | `feat(moe): zero-overhead activation storage via internal score weighting` |
| !649 | `[Add Test]: add UT and ST for trainer / trainer/utils / core/utils` |
| !656 | `feat: add distributed ops AllGatherMatmul and MatmulReduceScatter (MC2)` |
| !657 | `bugfix: add CMake pre-build support for mindspore custom_ops` |
| !659 | `feat(dtensor): add DTensor manual_seed (PyTorch parity)` |
| !660 | `feat(context-parallel): add DSA context parallel style` |
| !661 | `feat: add distributed op RotaryPositionEmbedding with UT and MindSpore ST` |
| !663 | `feat(context_parallel): add async colossal cp coverage` |
| !664 | `refactor ms pipeline_stage` |
| !665 | `test: improve UT coverage for core/shard and custom_ops/experimental` |
| !667 | `Restructure multicore parallel code` |
| !668 | `merge bugfix_strided_shard_master into master` |
| !669 | `feat(gate-doctor): autonomous PR gate doctor skill` |
| !670 | `feat(examples): MindSpore llama3 TP+FSDP demo and local embedding` |
| !671 | `feat: add pipeline parallel comm/compute overlap (B/F + lazy a2a)` |
| !673 | `disable_dispatch` |
| !675 | `merge master into master` |
| !677 | `test: add MindSpore PP+EP+overlap PoC test` |
| !678 | `merge dmodule_protocols_part1 into master` |
| !680 | `refactor(test): overhaul shard ST test structure and file naming` |
| !682 | `refactor(moe+ep): decouple token permutation and dispatch from forward` |
| !683 | `fix: avoid recompute packing sharded param views` |
| !684 | `[task][fully_shard] improve fully_shard ut cov rate` |
| !685 | `feat: register Detach and StopGradient distributed ops in element_wise_ops.yaml` |
| !687 | `feat(moe): add auxiliary load-balance loss, distributed expert_bias synchronization, and sequence_partition_group` |
| !689 | `fix(dtensor): keep parameter casts as dtensor` |
| !690 | `multicore megamoe rename w1_grad and w2_grad; add test cases` |
| !691 | `migrate distributed op` |
| !693 | `fix: package sapp nd memory estimation` <!-- codespell:ignore nd --> |
| !694 | `perf(pipeline): use platform.empty for recv buffer allocation` |
| !695 | `refactor: add dist-op-analysis & dist-op-dev skills; simplify dev workflow docs` |
| !697 | `refactor: preserve tensor identity in activation swap for keep_graph` |
| !699 | `merge feat-sapp-nd-pr2 into master` |
| !700 | `merge pp-swap into master` |
| !703 | `fix(dtensor): size dim_map by tensor rank in _calc_shard_info` |
| !704 | `fix: add loss parallel support for tensor parallel training` |
| !705 | `merge pp-swap into master` |
| !706 | `feat(examples): add Llama3 PP demos and unify training steps` |
| !707 | `[fix][fully_shard] adjust prefetch, unshard position` |
| !710 | `[feature][pipeline_parallel] pipelinestage support dx,dw compute` |
| !713 | `test(context-parallel): add cp unit coverage` |
| !720 | `feat(examples): add MoE Expert Parallel correctness demo` |
| !723 | `fix(shard): overlap MindSpore replicate allreduce` |
| !724 | `test(torch): add PP+FSDP+TP composite accuracy tests vs single-card baseline` |
| !725 | `bug_fix` |
| !726 | `feat: support tp dtensor local rewrap for context parallel` |
| !729 | `feat(pipeline): overlap_bf_recompute_fixes` |
| !730 | `merge fix-grad-norm into master` |
| !734 | `test: add comprehensive UT for core/activation_checkpoint (17.9% → 95% coverage)` |
| !737 | `feat(example): add PP(1F1B)+FSDP2+CP demo for Llama3 on Torch backend` |
| !738 | `chore: make agent tooling assistant-agnostic for any AI coding assistant` |
| !740 | `feat(moe): add PP+EP 1F1B correctness verification example` |
| !741 | `ms_support_recompute_independent_schedule` |
| !744 | `fix fsdp hook clear` |
| !749 | `merge fix/ms-fsdp-param-hook-migration into master` |
| !752 | `fix(init_weights): preserve parameter attributes when moving to target device` |
| !754 | `fix(activation_checkpoint): exclude shared functions from callable overlap tracking` |
| !755 | `merge sapp_ut_coverage1 into master` |
| !756 | `fix(dtensor): return self from MindSpore in-place random ops` |
| !757 | `[shard] Preserve DTensor identity for in-place ops on the dispatch bypass` |
| !759 | `[bugfix] fix FSDP device-mesh construction in the meta context for tp>1/dp=1 and noncontiguous compat all-reduce errors` |
| !766 | `test: fix process_group worker path and avoid double world-group teardown` |
| !767 | `fix dcp level st` |
| !768 | `feat(pipeline_parallel): forward-side P2P prefetch for overlap_b_f via batch_isend_irecv` |
| !770 | `fix dcp level st` |
| !771 | `fix(fsdp2): use assign=True when loading full state dict to fix optimizer device check on torch >= 2.8` |
| !775 | `fix(packaging): tag wheel per python/arch and unify cxx11 abi per backend` |
| !777 | `fix(codecheck): add missing docstrings for remaining C0116 warnings` |
| !779 | `fix(dtensor): refine MindSpore random op inplace dispatch` |
| !782 | `merge fix/ms-param-shape-error into master` |
| !783 | `muon, Hp` |
| !786 | `fix(pipeline): default PipelineStage device to current accelerator` |
| !787 | `fix(pipeline): reject batch size not divisible by micro_batch_num on MindSpore` |
| !788 | `test: reduce NPU level0 ST gate load with gloo level0 coverage` |
| !790 | `[feature][pipeline_parallel] torch backend pp support FSDP metastep` |
| !795 | `merge fix_named_parameters into master` |
| !803 | `fix: remove duplicate empty decorator in sparse_grad_kl_loss test` |
| !806 | `[bugfix] fix reduce type` |
| !808 | `Add MHC pre clamp sinkhorn custom op` |

---

## Known Limitations

- **MindSpore CP**: DSA-family async CP supports only the PyTorch backend; MindSpore uses no-op placeholders
- **DTensor**: centric communication and cross-mesh redistribution remain unimplemented
- **EP**: the basic dropless workflow, communication/computation overlap, hot expert migration, and replicas of heavily used experts remain unimplemented
- **PP**: ZBV and SeqPP scheduling, and assigning different device counts to individual PP stages, remain unimplemented
- **HyperOffload**: SAS/SPO/SAC user configuration interfaces, offload with memory semantics, and automatic strategy generation remain unimplemented
- **AutoParallel**: Fast-Tuner and PARADISE are currently demo features under continued optimization
- **One-sided communication**: AllToAll, AllReduce, ReduceScatter, and low-precision communication with high-precision accumulation remain unimplemented

---

## Upgrade Guide

To upgrade from v0.2.0 to v1.0.0:

1. **Activation Checkpoint API changes**: new `checkpoint_wrapper` / `swap_wrapper` / `swap_tensor_wrapper` interfaces; use these in place of the older APIs
2. **Optimizer module**: new `Muon` / `ChainedOptimizer` / `get_hyper_optimizer` / `get_hyper_lr_scheduler` interfaces can be integrated directly into training
3. **PP API extensions**: `PipelineStage` adds dx/dw computation and FSDP metastep support; `ScheduleInterleaved1F1B` adds `overlap_b_f` / `overlap_p2p` parameters
4. **CP API extensions**: new `AsyncContextParallel` and DSA-family interfaces
5. **Wheel packaging changes**: wheel filenames include Python ABI and host architecture tags; native adapters must match the target framework's
   CXX11 ABI contract

---

## Feedback and Contributions

- [Report an issue](https://gitcode.com/mindspore/hyper-parallel/issues)
- Contribute code: join the [Parallel Training System SIG](https://www.mindspore.cn/sig/Parallel%20Training%20System)
- License: Apache 2.0
