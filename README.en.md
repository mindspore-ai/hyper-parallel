# HyperParallel

An Ascend SuperPod-affinity distributed parallel acceleration library that simplifies supernode programming and unleashes computational potential.

HyperParallel provides Ascend SuperPod-affinity distributed parallel acceleration capabilities. Whilst maintaining ease of use, it addresses the architectural characteristics of Ascend SuperPods, including resource pooling, peer-to-peer architecture, hierarchical and diverse network topology, and FP8 low-precision formats. It implements distributed parallelism from cluster level to multi-core parallelism within chips, supports unified pooled management of CPU DRAM and NPU HBM, topology-aware scheduling and communication path planning, and FP8 mixed-precision training amongst other Ascend SuperPod-affinity acceleration capabilities.  
Key design principles:  
**Decoupling of Model and System Optimisation**: With the continuous evolution of LLM and multimodal algorithm architectures, performance optimisation techniques have also been advancing. The traditional integrated architecture of algorithm and system optimisation poses challenges for algorithm iteration and long-term system maintenance. Through HyperParallel, we aim to evolve the programming model from system optimisation embedded within model scripts to a decoupled model and system optimisation approach, with implicit injection of parallelism, recomputation, offload and other system optimisations. We support the evolution of parallel paradigms from SPMD to MPMD, further supporting coordinated optimisation of cluster MPMD and multi-core MPMD. We support the evolution of compute-storage relationships from Stateful to Stateless with separated computation and state. This supports large language model training, multimodal large model training, and reinforcement learning capabilities.  
**End-to-End Determinism**: To further ensure training stability and precision reproducibility, HyperParallel supports end-to-end determinism, including high-performance deterministic computation, communication, data preprocessing, and random number determinism, supporting floating-point bitwise alignment. All supported models are validated using determinism. Although there is some performance degradation, we still recommend enabling determinism during training for precision reproducibility, rapid SDC detection, and bug identification.  
**Unified Training and Inference**: As Reasoning RL and Agentic RL tasks become increasingly complex, the training-inference inconsistency problem causing reinforcement learning training convergence difficulties has become more prominent. HyperParallel will explore a unified training-inference architecture, achieving performance optimisation for both training and inference through a single acceleration framework, strengthening training-inference consistency and ensuring RL convergence.  
**Hybrid Dynamic-Static Execution**: Optimisation based on static graphs is an important means of further improving performance. For instance, capabilities such as compute-communication concurrency, memory analysis, and execution sequence orchestration based on static graphs can effectively optimise performance, which are not easily achievable in dynamic graph mode. However, dynamic-to-static compilation support is extremely challenging, and complete dynamic-to-static conversion is not yet achievable. HyperParallel will support partial dynamic-to-static conversion through certain syntax constraints, utilising MindSpore's advanced graph optimisation capabilities to further enhance performance.

**This project is under rapid iteration, and related features will continue to be open-sourced. Contributions are welcome.**

## Architecture Overview

<div align="center"> <img src="./docs/images/HyperParallel.jpg" width = 60% /> </div>

### HyperShard: Programming Model Evolution, System Optimisation Embedded in Model → Decoupled Model and System Optimisation

- SuperPod Layout: Unified modelling of tensor sharding, device mapping, and communication paths, achieving single-card abstraction for SuperPods;
- Declarative HSDP/TP/CP/EP: Implicit injection of optimisations such as parallelism, recomputation, and offload into models, achieving decoupling of model code and system optimisation code, improving algorithm development efficiency;

### HyperMPMD: Parallel Paradigm Evolution, SPMD → Cluster MPMD → Cluster + Multi-Core MPMD

- Distributed MPMD: Supports heterogeneous model sharding, supports arbitrary device allocation for model slices;
- Multi-Core MPMD: Intra-chip multi-core MPMD parallelism, combined with core-level memory semantic one-sided communication, enhancing compute-communication overlap and MAC utilisation;  

### HyperOffload: Compute-Storage Relationship Evolution, Stateful → Stateless Computation-State Separation

- Unified Programming for Remote and Local Tensors: Supports tensor location allocation, hides remote data transfer, improves cluster memory utilisation;  
- Remote Tensor Prefetching and Caching, Full Model Offload: DDP/HSDP+Offload replaces complex parallel modes such as DP/TP/PP/CP/SP/EP, simplifying system design and improving performance;

## Key Features

- Models
    - [x] DeepSeekV3
    - [ ] DeepSeekV3.2
    - [ ] Qwen3
    - [ ] Qwen3-MoE
    - [ ] Qwen3-VL
    - [ ] Qwen3-Omni

- HyperShard
    - DTensor
        - [x] DTensor basic
        - [x] DTensor redistribute
        - [ ] DTensor centric communication
        - [ ] Cross Mesh DTensor redistribution
    - HSDP
        - [x] Parameter & Optimiser Sharding
        - [x] Parameter & Optimizer & Gradient Sharding
        - [x] Overlap
        - [ ] Dynamic-to-Static Conversion
    - Shard
        - [x] Distributed Operator Support List (to be supplemented)
        - [x] Custom Distributed Operator Registration
        - [x] Custom Shard, supports user-defined parallel integration into DTensor parallel workflow
        - TP
            - [x] 1D
            - [ ] Higher-Dimensional TP, 2D/2.5D/3D
        - EP
            - [ ] Dropless Basic Workflow
            - [ ] Compute-Communication Overlap
            - [ ] Expert Hot Migration, addressing load imbalance
            - [ ] Hot Expert Replication, addressing load imbalance
        - CP
            - [ ] DeepSpeed Ulysses
            - [ ] Ring Attention
            - [ ] 3D Sequence Parallelism
        - [ ] Overlap
    - Distributed Random Numbers
        - [ ] Random Number Generation
        - [ ] DropOut

- HyperMPMD
    - Pipeline Parallelism
        - [x] Gpipe
        - [x] 1F1B
        - [x] VPP
        - [ ] ZBV
        - [ ] SeqPP
        - [ ] Overlap
        - [ ] Different Device Allocation per PP Stage
    - Subgraph Partitioning
        - [ ] Multimodal Encoder/Decoder Partitioning to Different Devices
    - Multi-Core Parallelism
        - [ ] Multi-Core Parallelism - O0: Through framework-level host CPU scheduling, supports cube, vector, and one-sided communication operator execution across cores;
        - [ ] Multi-Core Parallelism - O1: Scheduling offloaded to AICore, supports cube, vector, and one-sided communication operator execution across cores, further improving performance;
        - [ ] MoE Compute-Communication Overlap Optimisation Based on Multi-Core Parallelism
        - [ ] PP 1B1F Compute-Communication Overlap Optimisation Based on Multi-Core Parallelism

- HyperOffload
    - [ ] SAS (Selective Activation Swap): Supports user-configurable arbitrary activation swap to host CPU DRAM;
    - [ ] SPO (Selective Parameter/Gradient/Optimizer Offload): Supports user-configurable arbitrary P/G/O offload to host CPU DRAM;
    - [ ] SAC (Selective Activation Checkpointing): Supports user-configurable arbitrary recomputation scope;
    - [ ] Activation Swap and Checkpointing Coordinated Configuration
    - [ ] Memory Semantic-Based Offload
    - [ ] Automatic Activation Swap Strategy Generation

- AutoParallel
    - [x] Fast-Tuner: Based on profiling information, constructs black-box cost models, automatically generates multi-dimensional hybrid parallel strategies through enumeration, pruning, and search. Currently a demo feature, still under continuous optimisation;
    - [ ] PARADISE: Based on Symbolic cost model to simulate memory and compute-communication costs, efficiently generates optimal multi-dimensional hybrid parallel strategies. Currently a demo feature, still under continuous optimisation;

- One-Sided Communication
    - [ ] Symmetry Memory
    - [ ] AllToAll
    - [ ] AllGather
    - [ ] AllReduce
    - [ ] ReduceScatter
    - [ ] Low-Precision Communication with High-Precision Accumulation

- Fault Recovery
    - [ ] DCP (Distributed Checkpoint)
        - [ ] Huggingface Format Support
        - [ ] Different Sharding Strategy Conversion Support
    - [ ] Basic Fault Recovery Workflow
    - [ ] Process-Level Fast Recovery
    - [ ] Last Words, Fault-Triggered Checkpoint Saving, saving checkpoint at fault step for zero rollback;
    - [ ] SDC Detection

- Tools
    - Precision Monitoring
        - [ ] global norm
        - [ ] local norm
        - [ ] local loss

    - DryRun
        - [ ] Memory Overhead Analysis
        - [ ] Single-Card Cluster Execution Simulation

## Installation Guide

Currently only installation from source is supported. You need to execute:

```bash
git clone https://gitcode.com/mindspore/hyper-parallel.git
cd hyper-parallel
pip install .
```

HyperParallel depends on a deep learning framework. Before using HyperParallel, you need to:

- Install a deep learning framework
- Recommended MindSpore version >= 2.8, preferably using the latest MindSpore version, refer to [here](https://atomgit.com/mindspore/mindspore#%E5%AE%89%E8%A3%85)

## Usage Instructions

1. Use hsdp for data parallelism or zero sharding optimisation

```python
from hyper_parallel import hsdp

# Configure data parallelism
model = hsdp(model, shard_size=1)

# Or configure zero sharding
model = hsdp(model, shard_size=dp_size, optimizer_level="level1")
```

2. Use shard for tensor parallelism

```python
from mindspore.nn.utils import no_init_parameters
from hyper_parallel import DTensor, Layout, hsdp, init_parameters, shard

# Define tensor layout
layout = Layout((dp, mp), ("dp", "mp"))
x_layout = layout("dp", "mp")
w_layout = layout("mp", "None")
out_layout = layout()

# Delayed network weight initialisation
with no_init_parameters():
    model = SimpleModel()

# Configure sharding for network input/output/weights
sharding_plan = { "forward": { "input": (x_layout,), "output": (out_layout,)},
                "parameter": {"weight": w_layout}}
model = shard(model, sharding_plan)

# Can further configure hsdp
model = hsdp(model, shard_size=hsdp_shard_size, threshold=0)

# Sharded weight initialisation
model = init_parameters(model)

# Execute
x = DTensor.from_local(local_x, x_layout)
run_model(x, model)
```

3. Use PipelineStage and PipelineSchedule for pipeline parallelism

```python
from hyper_parallel import PipelineStage, Schedule1F1B

# Wrap the partitioned module into PipelineStage
stage = PipelineStage(splited_model, stage_index, stage_num=4)

# Select pipeline parallel scheduling
schedule = Schedule1F1B(stage, micro_batch_num=8)

# Execute
x = DTensor.from_local(local_x, x_layout)
schedule.run(x)
```

## Contributing

1. Fork this repository
2. Create a new Feat_xxx branch
3. Commit your code
4. Create a new Pull Request

If you have any suggestions for HyperParallel, please contact us through issues and we will address them promptly.
If you are interested in HyperParallel's technology or would like to contribute code, you are welcome to join the [Parallel Training System SIG](https://www.mindspore.cn/sig/Parallel%20Training%20System).

## License

[Apache 2.0 License](LICENSE)
