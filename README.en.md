# HyperParallel

An Ascend SuperPod-affinity distributed parallel acceleration library that simplifies supernode programming and
unleashes computational potential.

HyperParallel provides Ascend SuperPod-affinity distributed parallel acceleration capabilities. Whilst maintaining ease
of use, it addresses the architectural characteristics of Ascend SuperPods, including resource pooling, peer-to-peer
architecture, hierarchical and diverse network topology, and FP8 low-precision formats. It implements distributed
parallelism from cluster level to multi-core parallelism within chips, supports unified pooled management of CPU DRAM
and NPU HBM, topology-aware scheduling and communication path planning, and FP8 mixed-precision training amongst other
Ascend SuperPod-affinity acceleration capabilities.

Key design principles:

**Decoupling of Model and System Optimisation**: With the continuous evolution of LLM and multimodal algorithm
architectures, performance optimisation techniques have also been advancing. The traditional integrated architecture of
algorithm and system optimisation poses challenges for algorithm iteration and long-term system maintenance.
HyperParallel supports evolving the programming model from system optimisation embedded within model scripts to a
decoupled model and system optimisation approach, with implicit injection of parallelism, recomputation, offload and
other system optimisations. It supports the evolution of parallel paradigms from SPMD to MPMD, further supporting
coordinated optimisation of cluster MPMD and multi-core MPMD. It supports the evolution of compute-storage relationships
from Stateful to Stateless with separated computation and state, as well as large language model training, multimodal
large model training, and reinforcement learning capabilities.

**End-to-End Determinism**: To further ensure training stability and precision reproducibility, HyperParallel supports
end-to-end determinism, including high-performance deterministic computation, communication, data preprocessing, and
random number determinism, supporting floating-point bitwise alignment. All supported models are validated using
determinism. Although there is some performance degradation, enabling determinism during training remains recommended
for precision reproducibility, rapid SDC detection, and bug identification.

**Unified Training and Inference**: As Reasoning RL and Agentic RL tasks become increasingly complex, the
training-inference inconsistency problem causing reinforcement learning training convergence difficulties has become
more prominent. HyperParallel will explore a unified training-inference architecture, achieving performance optimisation
for both training and inference through a single acceleration framework, strengthening training-inference consistency
and ensuring RL convergence.

**Hybrid Dynamic-Static Execution**: Optimisation based on static graphs is an important means of further improving
performance. For instance, capabilities such as compute-communication concurrency, memory analysis, and execution
sequence orchestration based on static graphs can effectively optimise performance, which are not easily achievable in
dynamic graph mode. However, dynamic-to-static compilation support is extremely challenging, and complete
dynamic-to-static conversion is not yet achievable. HyperParallel will support partial dynamic-to-static conversion
through certain syntax constraints, utilising MindSpore's advanced graph optimisation capabilities to further enhance
performance.

## Architecture Overview

<div align="center"> <img src="./docs/images/HyperParallel.jpg" width = 60% /> </div>

### HyperShard: Programming Model Evolution, System Optimisation Embedded in Model -> Decoupled Model and System Optimisation

- SuperPod Layout: Unified modelling of tensor sharding, device mapping, and communication paths, achieving single-card
  abstraction for SuperPods;
- Declarative HSDP/TP/CP/EP: Implicit injection of optimisations such as parallelism, recomputation, and offload into
  models, achieving decoupling of model code and system optimisation code, improving algorithm development efficiency;

### HyperMPMD: Parallel Paradigm Evolution, SPMD -> Cluster MPMD -> Cluster + Multi-Core MPMD

- Cluster MPMD: Supports heterogeneous model sharding, supports arbitrary device allocation for model slices;
- Multimodal MPMD (Mpipe): Supports heterogeneous scheduling for multimodal pipeline parallelism, unlocking SuperPod
  utilisation for MLLMs;
- Intra-Chip Multi-Core MPMD: Intra-chip multi-core MPMD parallelism, combined with core-level memory semantic one-sided
  communication, enhancing compute-communication overlap and MAC utilisation;

### HyperOffload: Compute-Storage Relationship Evolution, Stateful -> Stateless Computation-State Separation

- Unified Programming for Remote and Local Tensors: Supports tensor location allocation, hides remote data transfer,
  improves cluster memory utilisation;
- Remote Tensor Prefetching and Caching, Full Model Offload: DDP/HSDP+Offload replaces complex parallel modes such as
  DP/TP/PP/CP/SP/EP, simplifying system design and improving performance;

## Key Features

- Models
    - [x] DeepSeekV3
    - [x] Qwen3.5-0.8B-Base
    - [x] Qwen3.5-35B-A3B-Base (MoE)
    - [x] Qwen3-VL-30B-A3B-Instruct (MoE)
    - [ ] DeepSeekV3.2
    - [ ] Qwen3-Omni

- HyperShard
    - DTensor
        - [x] DTensor basic
        - [x] DTensor redistribute
        - [x] manual_seed (distributed random number seed control)
        - [ ] DTensor centric communication
        - [ ] Cross Mesh DTensor redistribution
    - HSDP / FSDP
        - [x] Parameter & Optimiser Sharding
        - [x] Parameter & Optimiser & Gradient Sharding
        - [x] Overlap (full overlap mode)
        - [x] gradient_scaling_factor
        - [ ] Dynamic-to-Static Conversion
    - Shard / TP
        - [x] Distributed Operator Support List
        - [x] Custom Distributed Operator Registration (YAML registry + Python impl)
        - [x] Custom Shard
        - [x] DFunction (custom distributed autograd functions)
        - [x] parallelize_value_and_grad
        - [x] Loss Parallel (TP training loss parallelism)
        - TP Styles
            - [x] ColwiseParallel / RowwiseParallel / SequenceParallel
            - [x] parallelize_module (declarative TP interface)
            - [x] 1D
            - [ ] Higher-Dimensional TP, 2D/2.5D/3D
        - EP
            - [x] ExpertParallel / ExpertTensorParallel (basic workflow)
            - [x] MoE building blocks (GroupedExperts / TokenChoiceTopKRouter / MoE)
            - [x] Load balancing (expert_bias + aux_loss + AutoScaler)
            - [x] MoE zero-overhead activation storage
            - [x] MoE+EP token dispatch decoupling
            - [ ] Dropless basic workflow
            - [ ] Compute-Communication Overlap
            - [ ] Expert Hot Migration / Hot Expert Replication
        - CP
            - [x] ContextParallel (basic context parallelism)
            - [x] AsyncContextParallel (async context parallelism)
            - [x] DSA series (Indexer / Loss / SparseAttention)
            - [x] TP DTensor local rewrap
            - [ ] DeepSpeed Ulysses
            - [ ] Ring Attention
            - [ ] 3D Sequence Parallelism
        - [ ] Overlap
    - Distributed Random Numbers
        - [x] manual_seed (seed control)
        - [ ] DropOut

- HyperMPMD
    - Pipeline Parallelism
        - [x] GPipe
        - [x] 1F1B
        - [x] VPP (ScheduleInterleaved1F1B)
        - [x] PP+FSDP (MetaStep integration)
        - [x] PipelineStage dx/dw computation
        - [x] Compute-Communication Overlap overlap_b_f (CommComputeOverlap dual-thread orchestrator)
        - [x] batched P2P transport / overlap_p2p
        - [x] PP Activation Swap
        - [x] variable-layer + mixed-recompute under overlap_b_f
        - [ ] ZBV
        - [ ] SeqPP
        - [ ] Different Device Allocation per PP Stage
    - Mpipe Multimodal Parallelism
        - [x] Mpipe Transpose Scheduling
        - [ ] Mpipe Data Reordering
    - Subgraph Partitioning
        - [ ] Multimodal Encoder/Decoder Partitioning to Different Devices
    - Multi-Core Parallelism
        - [ ] Multi-Core Parallelism - O0
        - [ ] Multi-Core Parallelism - O1
        - [x] MoE Compute-Communication Overlap Optimisation Based on Multi-Core Parallelism
        - [ ] PP 1F1B Compute-Communication Overlap Optimisation Based on Multi-Core Parallelism

- HyperOffload
    - [x] Activation Checkpoint (checkpoint / checkpoint_wrapper / CheckpointPolicy)
    - [x] Activation Swap (swap / swap_wrapper / swap_tensor_wrapper / SwapManager)
    - [x] Activation Swap and Checkpointing Coordinated Configuration
    - [x] Swap Fusion
    - [ ] SAS (Selective Activation Swap)
    - [ ] SPO (Selective Parameter/Gradient/Optimizer Offload)
    - [ ] Memory Semantic-Based Offload
    - [ ] Automatic Activation Swap Strategy Generation

- Optimizer
    - [x] AdamW
    - [x] Muon (momentum-based optimizer)
    - [x] ChainedOptimizer (Muon+AdamW chained combination)
    - [x] get_hyper_optimizer
    - [x] Sharded Optimizer (FSDP/HSDP integration)
    - [x] gradient scaling factor + clip_grad enhancements

- AutoParallel
    - [x] SAPP-ND: ND Search (memory estimation + performance estimation)
    - [x] SAPP-PPB: Pipeline Parallelism Balancing
    - [ ] SAPP-Omni

- One-Sided Communication
    - [x] Symmetric Memory
    - [x] AllGather
    - [x] AllGatherMatmul / MatmulReduceScatter (MC2 fused communication ops)
    - [ ] AllToAll
    - [ ] AllReduce
    - [ ] ReduceScatter
    - [ ] Low-Precision Communication with High-Precision Accumulation

- Fast Fault Recovery
    - [x] DCP (Distributed Checkpoint)
        - [x] Distributed checkpoint save/load
        - [x] Async staging save
        - [x] Offline format transform
        - [ ] Huggingface Format Support
        - [ ] Different Sharding Strategy Conversion Support
    - [ ] Basic Fault Recovery Workflow
    - [ ] Process-Level Fast Fault Recovery
    - [ ] Last Words (fault-triggered checkpoint saving)
    - [ ] SDC Detection

- Trainer
    - [x] LLMTrainer framework
    - [x] VLTrainer framework (visual-language)
    - [x] Callbacks (Logging / MoeMonitor)
    - [x] parallel_dims configuration

- Integration
    - [x] LlamaFactory integration (activation recompute & swap + HSDP)

- Tools
    - Precision Monitoring
        - [ ] global norm
        - [ ] local norm
        - [ ] local loss

    - DryRun
        - [ ] Memory Overhead Analysis
        - [ ] Single-Card Cluster Execution Simulation

## Installation Guide

HyperParallel offers two installation methods:

- **pip installation**: install an already built `hyper-parallel` package and use extras to select runtime deep
  learning framework dependencies.
- **source build**: use `./build.sh` to generate a whl package, and use build arguments to decide whether native
  extensions are compiled.

If you only need to install a released package, prefer `pip install`. If you need to generate a whl package locally or
customize native extension build options, build from source.

### 1. Install With pip

`pip install` extras only control Python runtime dependencies. They do not control native extension compilation.

| Command                                   | Installed dependencies                                          | Use case                                                                          |
|-------------------------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------------------------|
| `pip install hyper-parallel`              | Common dependencies only, no deep learning framework            | You manage framework versions yourself or only use framework-independent features |
| `pip install 'hyper-parallel[mindspore]'` | Common dependencies + `mindspore>=2.10`                         | Use a supported MindSpore backend                                                 |
| `pip install 'hyper-parallel[torch]'`     | Common dependencies + `torch==2.9.1` + `torch-npu==2.9.1`       | Use the default PyTorch 2.9 backend                                               |
| `pip install 'hyper-parallel[torch26]'`   | Common dependencies + `torch==2.6.0` + `torch-npu==2.6.0.post3` | Use the PyTorch 2.6 backend                                                       |
| `pip install 'hyper-parallel[torch27]'`   | Common dependencies + `torch==2.7.1` + `torch-npu==2.7.1`       | Use the PyTorch 2.7 backend                                                       |
| `pip install 'hyper-parallel[torch29]'`   | Common dependencies + `torch==2.9.1` + `torch-npu==2.9.1`       | Explicitly use the PyTorch 2.9 backend                                            |
| `pip install 'hyper-parallel[all]'`       | Common dependencies + MindSpore + default PyTorch 2.9           | Use both backends in the same environment                                         |

In shells such as zsh, quote package names with extras so `[]` is not treated as a glob pattern.

### 2. Build a Wheel From Source

Building hyper-parallel from source can compile three optional native modules: `multicore`, `symmetric memory`, and
`custom ops`. The indexed Dataset C++ helper is a required wheel artifact and is built on every `build.sh` invocation.

Building a whl with `build.sh` supports the following build arguments:

| Argument       | Default     | Values                                                                  | Description                                                                                                                   |
|----------------|-------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `--multicore`  | `all`       | `off`, `mindspore`, `ms`, `torch`, `pytorch`, `all`, `both`             | Controls the multicore build scope; `ms` is equivalent to `mindspore`, `pytorch` is equivalent to `torch`, and `both` is equivalent to `all` |
| `--shmem`      | `all`       | `off`, `mindspore`, `ms`, `torch`, `pytorch`, `all`, `both`             | Controls the symmetric memory build scope; `all` builds the common library, MindSpore wrapper, and PyTorch wrapper             |
| `--custom-ops` | `on`        | `on`, `off`                                                             | Enables or disables the MindSpore custom-ops build                                                                            |
| `--soc-list`   | `ascend910b,ascend910_93` | Comma-separated selection of `ascend910b`, `ascend910_93`, and `ascend950` | Selects packaged kernels; `ascend910b` (910B) and `ascend910_93` (910C) are supported; `ascend950` reports an optional failure |
| `--strict`     | `off`       | `on`, `off`                                                             | `off` retains the wheel with a structured warning; use `on` for an explicitly strict developer build                          |
| `--jobs`       | `nproc`     | Positive integer                                                        | Sets native compilation parallelism                                                                                            |
| `--clean`      | disabled    | Flag                                                                    | Rebuilds selected component work/install outputs while retaining downloaded dependencies                                      |

Source build environment requirements for hyper-parallel are as follows:

| Environment item                | Requirement                                                               | Notes                                                                                                             |
|---------------------------------|---------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| Python                          | 3.10, 3.11, or 3.12                                                       | The built whl can only be installed into the matching Python minor version                                        |
| Python build packages           | `setuptools`, `wheel`, and `pybind11`                                     | `pybind11` and the active Python development headers are required by the indexed Dataset helper                   |
| Host C/C++ toolchain            | GCC/G++ >= 7.3.0 with C++17 support                                       | GCC/G++ 7.3.0--11.3.0 is recommended; newer versions emit a warning                                               |
| Host architecture               | `aarch64` or `x86_64`                                                     | Each whl targets one host architecture and one CPython ABI                                                         |
| CMake                           | >= 3.18                                                                   | Required for native extension builds                                                                              |
| Linux build tools               | GNU Make, Git, binutils, coreutils, `tar`, `sed`, and `awk`               | Required by dependency preparation, ELF validation, and the CANN operator build pipeline                          |
| CANN toolkit and ops packages   | >= 9.1.0 complete development environment                                 | Source the selected CANN `set_env.sh`; it must provide `bisheng`, `asc_opc`, headers, `libopapi.so`, and `ops_base` |
| Ninja                           | Available on `PATH` for MindSpore native targets                          | Required by `CustomOpBuilder` builds                                                                               |
| MindSpore                       | >= 2.10                                                                   | Required when `--custom-ops on`, `--multicore mindspore/all`, or `--shmem all/mindspore`                          |
| PyTorch and NPU adapter package | Backend-compatible pair with `_GLIBCXX_USE_CXX11_ABI=1`                   | Required when `--multicore torch/all`, or `--shmem all/torch`; the build uses the pair installed in the active environment |

```bash
git clone https://gitcode.com/mindspore/hyper-parallel.git
cd hyper-parallel

# The default CANN installation is sourced automatically when needed. Source a
# custom installation explicitly before build.sh.
./build.sh
./build.sh --multicore all --shmem all --custom-ops on --soc-list ascend910b,ascend910_93
./build.sh --multicore torch --shmem torch --strict off
./build.sh --multicore off --shmem off --custom-ops off
# Install the exact wheel path printed by build.sh.
wheel_path=/absolute/path/printed/by/build.sh
pip install "${wheel_path}"
```

Every `build.sh` invocation freshly assembles `build/native/payload/hyper_parallel` from component install roots and
creates a wheel and prints its exact path. PYTHONPATH development uses that same payload. A successful component
script refreshes its own payload slice for focused incremental builds. Heavy SHMEM and per-SoC vendor caches are retained by default; lightweight
framework adapters are rebuilt from clean framework-identity work directories on every invocation. Use `--clean` to
rebuild all work for the selected components. A matching dependency cache is reused automatically; absent or
inconsistent locked dependencies are downloaded/refreshed.

For a multi-SoC multicore build, HyperParallel builds the HyperMegaMoe vendor for every selected kernel target and
combines the resulting kernel/config trees into one package. The package carries one common host payload after the
vendor inputs and host ABI have been checked for consistency.

> Note: the built whl has requirements on the glibc version of the runtime environment. The glibc version in the
> installation environment must be no lower than the glibc version in the build environment.
> If you need to deploy to an older system, build inside an older release image. For example, a whl built on OpenEuler
> 22.03 (glibc 2.34) cannot run in an environment with glibc < 2.34.
> Release wheels use the glibc baseline selected by the release environment. The resulting ELF payload determines the
> required runtime glibc floor.

Native source builds and prebuilt release wheels require CANN 9.1.0 or newer. Source the selected CANN `set_env.sh`
before building; the build reads the exported `ASCEND_HOME_PATH`. The default CANN path is activated automatically.

### 3. Activate the Multicore Custom OPP Environment

Both an installed wheel and a PYTHONPATH development build require the packaged multicore `set_env.bash` before the
application or framework Python process starts. The script activates the adjacent CANN custom OPP vendor for that shell.

For a PYTHONPATH development build:

```bash
source /usr/local/Ascend/cann/set_env.sh
export PYTHONPATH=/path/to/hyper-parallel:${PYTHONPATH:-}
source /path/to/hyper-parallel/build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
python application.py
```

Start the application or framework Python process after sourcing the payload script.

For an installed wheel:

```bash
source /usr/local/Ascend/cann/set_env.sh
source "$(command -v hyper_parallel_multicore_set_env.bash)"
python application.py
```

After wheel installation, source the locator installed in the active Python environment's `bin` directory.

Run the script before importing MindSpore or torch/torch_npu. Missing activation reports
`HP-NATIVE-OPP-NOT-ACTIVATED`; detecting it after a framework import reports
`HP-NATIVE-OPP-ACTIVATION-TOO-LATE` and requires a new Python process.

## Quick Start

1. Use `fully_shard` for data-parallel parameter sharding

```python
from hyper_parallel import fully_shard, init_device_mesh

mesh = init_device_mesh(device_type="npu", mesh_shape=(dp_size,), mesh_dim_names=("dp",))
model = fully_shard(model, mesh=mesh)
```

2. Use `shard_module` for tensor parallelism

```python
from mindspore.nn.utils import no_init_parameters
from hyper_parallel import DTensor, fully_shard, init_device_mesh, init_parameters, shard_module
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.sharding_plan import ShardingPlan

# Define device mesh and placement
mesh = init_device_mesh(device_type="npu", mesh_shape=(dp_size, tp_size), mesh_dim_names=("dp", "tp"))
x_placement = (Shard(0), Shard(1))
w_placement = (Replicate(), Shard(0))
out_placement = (Shard(0), Replicate())

# Delayed network weight initialization
with no_init_parameters():
    model = SimpleModel()

# Configure sharding for network input/output/weights
sharding_plan = ShardingPlan(
    input_plan={"input": x_placement},
    output_plan={"output": out_placement},
    plan={"weight": w_placement},
)
model = shard_module(model, device_mesh=mesh, sharding_plan=sharding_plan)

# Can further configure fully_shard
model = fully_shard(model, mesh=mesh["dp"])

# Sharded weight initialization
model = init_parameters(model)

# Execute
x = DTensor.from_local(local_x, mesh, x_placement)
run_model(x, model)
```

3. Use declarative TP Styles for tensor parallelism

```python
from hyper_parallel import ColwiseParallel, RowwiseParallel, parallelize_module, init_device_mesh

tp_mesh = init_device_mesh("npu", (tp_size,), mesh_dim_names=("tp",))

parallelize_module(
    model,
    tp_mesh,
    {
        "attn.q_proj": ColwiseParallel(),
        "attn.k_proj": ColwiseParallel(),
        "attn.v_proj": ColwiseParallel(),
        "attn.o_proj": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    },
)
```

4. Use `PipelineStage` and `PipelineSchedule` for pipeline parallelism

```python
from hyper_parallel import DTensor
from hyper_parallel.core.pipeline_parallel import PipelineStage, Schedule1F1B

# Wrap the partitioned module into PipelineStage
stage = PipelineStage(split_model, stage_index, stage_num=4)

# Select pipeline parallel scheduling
schedule = Schedule1F1B(stage, micro_batch_num=8)

# Execute
x = DTensor.from_local(local_x, input_mesh, input_placements)
schedule.run(x)
```

5. Use Optimizer for Muon+AdamW chained optimisation

```python
from hyper_parallel.core.optimizer import get_hyper_optimizer

optimizer = get_hyper_optimizer(
    model=model,
    muon_params=muon_param_groups,
    adamw_params=adamw_param_groups,
    muon_kwargs={"lr": 0.02, "momentum": 0.95},
    adamw_kwargs={"lr": 3e-4, "weight_decay": 0.1},
)
```

6. Use Activation Checkpoint/Swap for memory optimisation

```python
from hyper_parallel.core.activation_checkpoint import checkpoint_wrapper, swap_wrapper

model.layers[0] = checkpoint_wrapper(model.layers[0])
model.layers[1] = swap_wrapper(model.layers[1])
```

7. Use MoE Compute-Communication Overlap Optimisation Based on Multi-Core Parallelism

For details, see the [MoE Multi-Core Parallelism Guide](./docs/guide/multicore_moe.md).

## Documentation

- [Docs Hub](./docs/index.md) - Documentation index and navigation
- [Installation Guide](docs/installation.md) - Source build, dependencies
- [Feature Guides](./docs/guide/) - 10 core feature usage guides
- [API Reference](./docs/api/api_reference.md) - Interface descriptions organised by feature module
- [FAQ & Troubleshooting](./docs/faq.md) - Common issues and solutions
- [AI-Assisted Development](./AGENTS.md) - AI-assisted development capabilities
- [Contributing](./docs/contributing/) - Dev environment, testing, release process
- [Release Notes](./hyper_parallel_v1.0.0_release_notes.md) - Version change records

## Contributing

1. Fork this repository
2. Create a new Feat_xxx branch
3. Commit your code
4. Create a new Pull Request

If you have any suggestions for HyperParallel, please contact us through issues and we will address them promptly.
If you are interested in HyperParallel's technology or would like to contribute code, you are welcome to join
the [Parallel Training System SIG](https://www.mindspore.cn/sig/Parallel%20Training%20System).

## License

[Apache 2.0 License](LICENSE)
