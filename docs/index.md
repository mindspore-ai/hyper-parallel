# HyperParallel Documentation

**English** | [简体中文](index.zh.md)

Documentation for the distributed parallel acceleration library optimized for Ascend SuperPoDs.

## Getting Started

| Document | Description |
|----------|-------------|
| [Installation Guide](./installation.md) | Source builds, dependencies, and environment configuration |

## Feature Guides

| Document | Description |
|----------|-------------|
| [HSDP / FSDP Data Parallelism](./guide/fsdp.md) | fully_shard, HSDPModule, overlap, and gradient scaling |
| [TP Tensor Parallelism](./guide/tensor_parallel.md) | ColwiseParallel, RowwiseParallel, parallelize_module, and Loss Parallel |
| [PP Pipeline Parallelism](./guide/pipeline_parallel.md) | PipelineStage, schedules, overlap_b_f, PP+FSDP, P2P prefetch, and Mpipe |
| [CP Context Parallelism](./guide/context_parallel.md) | ContextParallel, AsyncContextParallel, and DSA variants |
| [KDA Context Parallelism](./guide/kda_context_parallel.md) | Configuration and memory usage for KDA P2P, AllGather, Ulysses, and hybrid strategies |
| [KDA Recursive Doubling and Ulysses Hybrid CP](./guide/kda_recursive_doubling.md) | RD state scans, TND reset pruning, and transition caching |
| [Linear Attention TND Context Parallelism](./guide/linear_attention_tnd_cp.md) | KDA / GDN document boundaries, configuration, state exchange, and numerical semantics |
| [EP Expert Parallelism](./guide/expert_parallel.md) | ExpertParallel, MoE building blocks, and load balancing |
| [Activation Checkpoint / Swap](./guide/activation_checkpoint.md) | checkpoint_wrapper, swap_wrapper, swap_tensor_wrapper, and combined configuration |
| [Optimizer](./guide/optimizer.md) | AdamW, Muon, ChainedOptimizer, and learning rate schedulers |
| [Chunk Loss](./guide/chunk_loss.md) | Chunked output projection and cross-entropy for large vocabularies; Trainer and Qwen3-MoE integration |
| [DCP Distributed Checkpoints](./guide/distributed_checkpoint.md) | Saving/loading checkpoints, asynchronous staging, and offline conversion |
| [Automatic Parallelism](./guide/auto_parallel.md) | SAPP-ND and SAPP-PPB |
| [LlamaFactory + HyperParallel SFT Training](./guide/llama_factory_hyperparallel_sft_guide.md) | SFT configuration, launch, numerical validation, and performance validation |
| [Model Integration Validation](./guide/trainer/model_integration_validation.md) | Adapter declarations, structural checks, module parity, numerical validation matrices, and evidence reports |

## API Reference

| Document | Description |
|----------|-------------|
| [API Reference](./api/api_reference.md) | Complete interface documentation organized by feature |
| [DFunction](./api/dfunction.md) | Custom distributed autograd functions for Torch |

## FAQ and Troubleshooting

| Document | Description |
|----------|-------------|
| [FAQ and Troubleshooting](./faq.md) | Installation, runtime, communication, memory, and debugging |

## Contributing

| Document | Description |
|----------|-------------|
| [Development Environment](./contributing/dev_environment.md) | Environment setup and dependencies |
| [Testing Standards](./contributing/testing.md) | Test framework, markers, and distributed tests |
| [Release Process](./contributing/release.md) | Version releases and documentation updates |

## Other Documentation

| Document | Description |
|----------|-------------|
| [Release Notes v1.0.0](../hyper_parallel_v1.0.0_release_notes.md) | Version history |
| [PP overlap_b_f Design](./guide/pipeline_parallel_overlap_b_f.md) | Detailed compute-communication overlap design |
| [TP Styles](./guide/tensor_parallel_styles.md) | Tensor parallel style documentation |
| [Agent Workflow](./agent_workflow.md) | AI agent workflow reference |
