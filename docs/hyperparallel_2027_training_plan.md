# HyperParallel 2027 Key Capabilities and Work Plan

**English** | [简体中文](hyperparallel_2027_training_plan.zh.md)

> Prepared: October 20, 2026
>
> Updated: October 26, 2026
>
> Status: Planning draft
>
> Basis: Discussions of 2027 capabilities, industry trends, work breakdown, and competitiveness metrics
>
> Trend evidence cutoff: October 26, 2026; assessments for 2027 are planning assumptions

## 1. Scope and Overall Approach

This plan focuses on HyperParallel pretraining and fine-tuning; inference and RL are currently excluded.
It develops four core capabilities—**usability and decoupling, parallelism for complex models, high availability at 100,000 devices, and a multicore parallel development framework**—using the structure **2027 trends → key challenges → technical competitiveness goals → concrete work**.

Responsibilities of the four capabilities:

- **Codegen**: Generate specialized training implementations from models, user requirements, parallel strategies, and hardware capabilities, supporting Agent RSI and user extensions.
- **Multidimensional parallelism**: Determine how complex models and dynamic workloads are distributed across devices, nodes, and SuperPoDs.
- **High-availability runtime**: Protect training state and preserve effective training progress through failures and recovery.
- **Multicore parallel development framework**: Express and schedule core-level computation and communication to improve execution performance and new-operator development efficiency.

Each capability has five work packages, for twenty in total. Numerical correctness, convergence quality, diagnosability, and performance regression control are shared delivery and acceptance requirements.

The 100,000-device target refers to a single training job's design scale. Design/simulation, hardware exercises at available scale, and joint validation on 100,000 physical devices are accepted separately.
Quantitative thresholds, model coverage, resources, and delivery schedules must be set at project initiation.

## 2. Overview of 2027 Trends and Core Capabilities

| Capability | 2027 trend assessment | Key challenges | Technical competitiveness goal | Main metrics |
| --- | --- | --- | --- | --- |
| Usability and decoupling | Models, strategies, and hardware continue evolving; code generation and explicit training graphs see broader adoption. Agent-driven generation, validation, and optimization need structured interfaces. | Coupled model/distributed implementations make adaptation and upgrades increasingly expensive as combinations grow; automated optimization and user customization are difficult to maintain. | Use Codegen to decouple model, strategy, and execution; support Agent RSI; generate a specialized training framework for each user, with straightforward extensions. | Time to first correct training, generation/validation success rate, iteration cost to target performance, and person-days for customization and upgrades. |
| Parallel competitiveness | Sparse MoE, long sequences, all-modality inputs, and world models converge, making computation and resource needs more dynamic. | Jointly handle irregular computation, communication hotspots, memory peaks, and cross-modality imbalance. | Build multidimensional parallelism and dynamic resource scheduling for complex models, increasing trainable scale and end-to-end efficiency within resource and quality constraints. | Capability limits under fixed resources, effective throughput, maximum EP/CP time ratios, peak memory, and scaling efficiency. |
| High availability for 100,000 devices | Frontier training reaches the 100,000-device scale; failure handling and recovery directly affect productive output. | Failure impact, full-recovery overhead, and coordination costs grow; membership changes must preserve consistency. | Build a hierarchical fault-tolerant runtime with failure-domain isolation, continued training on healthy portions, and recovery/rejoining of failed portions. | Effective training output, P95/P99 recovery time, failure impact, lost device-hours, and recovery correctness. |
| Multicore parallel development framework | Heterogeneous compute units and core-level communication/computation fusion create more scheduling opportunities; new models continually introduce operators and fusion patterns. | Handwritten core assignment, memory management, and synchronization are expensive; tuning, training integration, and migration across hardware generations take too long. | Build a programmable, composable, tunable multicore framework that improves both execution and operator development efficiency. | Time to first correct operator execution, target performance, and training integration; handwritten code, component reuse, migration cost, and training gains. |

These trends are planning assessments based on public information. Existing foundations, planned additions, and evidence are detailed below.

## 3. Capability One: Usability and Decoupling—Personalized Training Framework Generation for Agent RSI

### 3.1 Trends for 2027

**Public signals of decoupling model expression from execution.** Hugging Face Modular Transformers generates complete model implementations from modular source files, combining reuse with readable model files. The October 2026 TorchTitan update further emphasizes explicit training graphs, cross-module optimization, and graph-level debugging. These developments suggest that generation and compilation increasingly connect model development with high-performance execution.
References: [Modular Transformers](https://huggingface.co/docs/transformers/main/modular_transformers), [TorchTitan October 2026 update](https://pytorch.org/blog/torchtitan-update-new-sota-models-performance-at-scale-titanrl/).

**Recent RSI for Infra examples.** As of October 26, 2026, public work covers continuous agent optimization of operators, compilation, and infrastructure algorithms. The table distinguishes iterative implementation search, reuse of optimization experience, and improvements to the optimization agent itself, preserving each example's validation scope.

| Example and publication date | Optimization loop and reported results | Implications for HyperParallel and evidence limits |
| --- | --- | --- |
| [Meta KernelEvolve](https://engineering.fb.com/2026/04/02/developer-tools/kernelevolve-how-metas-ranking-engineer-agent-optimizes-ai-infrastructure/), 2026-04-02 | Connects generation, compilation, correctness checks, performance diagnosis, and search; accumulates cross-task experience and post-trains specialized models on optimization trajectories. Meta reports over 25% higher training throughput for an advertising model on MTIA and reduces operator optimization from weeks to hours. | Supports optimizing infrastructure, accumulating measured feedback, and improving subsequent optimization. Requires executable evaluations and an experience store; reported gains apply to the specified model, hardware, and internal workflow. |
| [Google AlphaEvolve infrastructure progress](https://deepmind.google/blog/alphaevolve-impact/), 2026-05-07; [cloud service general availability](https://cloud.google.com/blog/products/ai-machine-learning/alphaevolve-is-available-for-everyone), 2026-07-09 | Continuously optimizes code using seed programs, executable scoring functions, and evolutionary search. Reported applications include TPU design, cache replacement, and compiler optimization; the cloud service organizes problem definition, scoring, optimization, and application into a reusable workflow. | Express parallel strategies, scheduling, and component selection as searchable problems with quality constraints. This is an inferred development direction; the results do not directly validate HyperParallel strategy search or complete framework generation. |
| [PyTorch KernelAgent](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/), 2026-03-06 | Real-hardware profiling drives diagnosis, rewriting, validation, and remeasurement, with shared memory reusing experience. On 100 KernelBench L1 tasks on H100, it reports a 1.56× geometric-mean speedup over default torch.compile. | Agents need attributable hardware feedback and explicit stopping criteria. Results concern forward operators; the baseline uses static shapes with CUDA Graph disabled. Training forward/backward and end-to-end gains need separate validation. |
| [Intel Day-0 Kernel Bring-Up, PyTorch Conference 2026](https://pytorch.org/blog/open-research-tooling-optimization-at-pytorch-conference-north-america-2026/), agenda published 2026-09-16, talk scheduled for 10-20 | The official agenda describes staged agents and long-term memory connecting profiling, analysis, code generation, validation, repair, and evaluation, reporting a reduction from weeks of operator development to overnight runs. | Suggests making new-hardware adaptation a repeatable workflow. The cited evidence is an agenda abstract; specific tasks, resource budgets, and quality definitions need verification and do not establish this project's numerical targets. |

These examples cover different RSI for Infra stages: implementation optimization, experience accumulation, and using trajectories to improve optimization agents. KernelEvolve explicitly reports specialized model post-training; iterative search or shared memory alone in the other examples does not demonstrate recursive improvement in model capability.
The plan therefore prioritizes verifiable optimization environments, editable generated artifacts, and reusable experiment feedback, measuring artifact gains separately from improvements in optimization-process efficiency.

**Planning assessment.** Training infrastructure in 2027 must support both human development and agent-driven continuous optimization. Users' models, data, hardware, and goals must become executable, verifiable training implementations that remain editable.
Support for Agent RSI centers on a generation–validation–hardware evaluation–diagnosis/improvement–experience reuse loop.
Codegen supplies traceable, regenerable modification targets; multidimensional parallelism and the multicore framework supply composable optimization spaces. Unified feedback selects candidates and improves later search; tasks excluded from experience accumulation test transfer benefits.

### 3.2 Key Challenges

One model must support multiple parallel combinations, hardware platforms, and high-performance components. When these differences enter model source, upgrades affect communication, replacement modules, weight mappings, and numerical validation, with maintenance cost growing alongside combinations.
Agent optimization needs understandable contracts and attributable feedback; user extensions need clear execution boundaries and maintainable customization points.

Usability means model developers retain clear model expressions, distributed and hardware optimizations evolve independently, and generation reliably combines them into a training framework suited to each user.

### 3.3 Technical Competitiveness Goal

> Use Codegen to decouple models, parallel strategies, and execution implementations. Enable agents to generate specialized training frameworks from user requirements and improve them through correctness and performance feedback. Provide readable, editable generated code and extension interfaces to reduce customization and upgrade costs.

| Competitive capability | User value | Development focus |
| --- | --- | --- |
| Agent RSI support | Agents continuously generate, validate, measure, and improve, reusing experience in later iterations. | Structured model/strategy descriptions, traceable changes, automated correctness checks, attributable performance feedback, and effective experience reuse. |
| A specialized framework for each user | Generate suitable implementations and component combinations from the user's model, data, hardware, topology, and training objectives. | Extend model code generation to component assembly, parallel execution, and runtime configuration, reusing shared runtime and high-performance components. |
| Easy extensions | Users can understand, debug, and modify execution and integrate custom modules, losses, optimizers, and training logic. | Readable code, clear boundaries, stable extension points, and traceable customization that survives regeneration and upgrades. |

**Existing foundation.** Repository Codegen already embeds parallel contracts, boundary communication, and module replacement in generated artifacts; the runtime reads frozen plans and applies sharding. Work in 2027 extends coverage, personalized assembly, and engineering completeness.
Reference: [Codegen Guide](guide/codegen.md).

### 3.4 Concrete Work

| ID | Work package | Work | Deliverables |
| --- | --- | --- | --- |
| 1.1 | Unified model/execution generation contracts | Define computation semantics, parameter layouts, communication boundaries, training state, and high-performance component interfaces; support HF and custom models; define generation coverage and executable checks. | Unified descriptions and validation, component interfaces, model/strategy coverage matrix. |
| 1.2 | Personalized framework generation | Select components and strategies from training goals, models, data, hardware, and topology; generate models, component assembly, and runtime configuration; connect to training-graph compilation. | Reproducible user-specific artifacts, dependency/configuration manifests, representative recipes. |
| 1.3 | Agent RSI optimization loop | Expose generation, compilation, checks, hardware evaluation, and result-reading interfaces; provide structured errors, numerical results, and bottleneck information; constrain search by correctness gates, resource budgets, and stopping criteria; retain reproducible trajectories and successful/failed experience for retrieval and reuse. | Agent tools, executable evaluation workflows, experience store, replayable trajectories, and cost comparison reports. |
| 1.4 | Extensions and artifact lifecycle | Keep generated code readable/debuggable; expose component and training-logic extensions; track customizations; handle regeneration, upstream upgrades, compatibility, and caches. | Extension interfaces, customization preservation, diff review tools, upgrade workflows, and examples. |
| 1.5 | Generated correctness and efficiency validation | Compare original, generated, and optimized models for forward, backward, parameter updates, and convergence; establish baselines from requirements to first correct training, target performance, and upgrade maintenance. | Artifact validation workflows, development-efficiency benchmarks, and quality/performance comparisons. |

### 3.5 Acceptance Metrics

| Metric | Measurement and comparison |
| --- | --- |
| Time to first correct training | Elapsed time and person-days from complete agreed requirements/inputs to generated artifacts passing prescribed training correctness checks. |
| Agent generation/validation success rate | Tasks passing without human repair divided by total tasks within pre-agreed model, hardware, strategy, and resource budgets. |
| Iteration cost to target performance | Iterations, human effort, evaluation device-hours, and total time under identical numerical and performance targets. |
| Benefits from experience reuse | Fix the agent version, tools, hardware, and search budget; on held-out tasks, compare validation success, performance-target achievement, and iteration costs with/without the experience store. Evaluate agent-version gains separately. |
| Extension efficiency | Person-days and handwritten changes for representative custom-module, loss, optimizer, and training-flow tasks. |
| Upgrade maintenance cost | Changes, regression cost, and person-days for model/framework upgrades while preserving customizations and quality. |

## 4. Capability Two: Parallel Competitiveness for Sparse MoE, Long Sequences, All-Modal Inputs, and World Models

### 4.1 Trends for 2027

**Public signals.** Dynamic Context Parallelism work shows that equal token counts in variable-length text/video training can require different computation, causing DP waits and pipeline bubbles. MoE Parallel Folding uses different parallel configurations for attention and expert layers. NVIDIA's October 2026 Cosmos update further expands multimodal, multisensor processing and world-model training tools.
References: [Dynamic Context Parallelism](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/), [MoE Parallel Folding](https://arxiv.org/abs/2504.14960), [Cosmos October 2026 update](https://blogs.nvidia.com/blog/cosmos-open-train-evaluate-deploy-world-models/).

**Planning assessment.** Sparse MoE, long sequences, all-modality inputs, and world models will coexist in training jobs. Architecture, inputs, and objectives jointly determine computation, communication, and memory requirements, demanding shared cost models, strategy planning, and dynamic scheduling.

### 4.2 Key Challenges

| Scenario | Core challenges | Parallel development direction |
| --- | --- | --- |
| Sparse MoE | Uneven routing, irregular computation, All-to-All hotspots, and mismatch between expert parameter size and active computation. | Topology-aware EP, balanced expert placement/execution, coordinated dispatch/combine, and joint TP/CP/PP/FSDP optimization. |
| Long sequences | Growing activation/communication costs; causal, sparse, and hybrid attention have different partition semantics and compute distributions. | CP for different attention structures, jointly planned with variable-length packing, compute balancing, recomputation, and offload. |
| All modalities | Text/image/audio/video branches have different costs; shapes are dynamic and cross-modal alignment/loss semantics are complex. | Coordinated encoder/backbone parallelism, modality-aware data organization, branch/pipeline balancing, and correct alignment/loss computation. |
| World models | Video, spatiotemporal structures, multiple views, and action-conditioned prediction create larger activations and complex graphs. | Explore temporal, spatial, and view partitioning for target architectures; optimize parallelism and memory for diffusion, flow-matching, or autoregressive training. |

Fixed sample or token counts do not fully describe costs. Worst-case configuration reduces utilization; average-case configuration may exceed memory at peaks. Local operator gains may be canceled by communication waits, transfers, and scheduling overhead.

### 4.3 Technical Competitiveness Goal

> Build multidimensional parallelism for complex architectures and dynamic workloads, expanding trainable scale within resource and quality constraints while improving end-to-end efficiency and reusing optimizations across target models, hardware, and cluster scales.

### 4.4 Concrete Work

| ID | Work package | Work | Deliverables |
| --- | --- | --- | --- |
| 2.1 | Hardware, workload, and memory modeling | Measure target GPU/NPU compute, memory, and hierarchical interconnects; model how routing, sequence length, modality mix, and micro-batches affect compute, communication, and memory peaks. | Hardware/workload profiles, calibrated cost and peak-memory models, prediction-error reports. |
| 2.2 | Joint parallel/resource planning | Jointly choose DP/FSDP, TP, EP, CP, PP, rank mapping, recomputation, offload, and prefetch; consider intra-/inter-SuperPoD communication; map search results to executable plans. | Topology-aware strategies, resource-budgeted recipes, capacity/throughput tradeoffs. |
| 2.3 | Sparse MoE and long-sequence execution | Optimize expert placement, dispatch/combine, and hotspots; support CP for target attention structures; improve variable-length packing, bucketing, and load assignment to control communication and pipeline waits. | MoE/long-sequence paths, EP/CP balance reports, regression cases for extreme routing and length distributions. |
| 2.4 | All-modal and world-model training | Coordinate encoders/backbones and balance modality branches/stages; implement spatiotemporal partitioning and memory optimization for representative world models; validate sample alignment and loss semantics. | Architecture recipes, modality/spatiotemporal workload benchmarks, correctness/performance evidence. |
| 2.5 | End-to-end performance and capability limits | Increase model size, context length, and modality load under fixed resources; measure startup/compilation cost, steady-state throughput, and scaling; compare same-hardware alternatives and add release regression checks. | Trainable limits, performance baselines, regression workflows, reproducible comparisons. |

### 4.5 Acceptance Metrics

- Trainable model size, sequence length, resolution, video length, and modality combinations under fixed resources.
- Effective throughput, device-hours for fixed training volume, and scaling efficiency on identical hardware/data with identical quality requirements.
- Maximum EP/CP compute-time ratios, imbalance-induced wait fraction, peak memory, and tail performance.
- Behavior and operating limits under extreme lengths, expert skew, and near-capacity resource usage.

**Load-balance measurement:** For one layer, micro-batch, and parallel group, define maximum load ratio as `R(x) = max(x_i) / mean(x_i)` and coefficient of variation as `CV(x) = std(x_i) / mean(x_i)`.
Compute at that granularity before aggregating P50, P95, and worst cases. Mark zero-effective-load cases separately rather than dividing by a zero mean.

For MoE, distinguish expert routing distribution from EP-rank compute time. For CP, measure actual attention work and device time, not token counts alone. Use device-side timing, separate synchronization waits, and report forward/backward independently.
End-to-end benefits must include balancing's extra communication, resharding, and scheduling costs. Effective tokens exclude padding and replicated parallel counts.
Drop rates, numerical behavior, and convergence remain shared constraints; apparent balance must not come from reducing useful computation.

## 5. Capability Three: A High-Availability Runtime for 100,000-Device Training

### 5.1 Trends for 2027

**Public signals.** The 2026 FT-HSDP paper reports fault-tolerance experiments on approximately 98,000 H100s. It uses DP replicas as fault-tolerance units so other replicas continue training while a failed replica leaves and recovers. This provides public engineering evidence for local recovery and continued training near the 100,000-device scale.
Reference: [FT-HSDP paper](https://arxiv.org/html/2602.00277v1).

**Evidence limits.** Section 6.2 explicitly attributes the single, short full-scale experiment to resource cost and reports more extensive training-quality comparisons on 256 H100s. Section 6.1 notes that repeating the 100,000-device test after fixes requires waiting for resources. The evidence therefore supports feasibility of large-scale recovery mechanisms; extrapolating long-run behavior and training quality needs separate validation.
Reference: [FT-HSDP experiments and quality validation](https://arxiv.org/html/2602.00277v1#S6).

**Planning assessment.** More devices and longer runs increase failure exposure. Recovery, coordination, and state transfer become major efficiency costs. At 100,000 devices, high availability must be part of training protocols, communication, and runtime design, evaluated through effective output during sustained operation.

### 5.2 Key Challenges

- One device, process, or link failure can propagate across communication groups, leaving many healthy devices waiting.
- Resource replacement, communication initialization, state loading, and first execution can create long recovery tails.
- Departing/rejoining members affect gradient aggregation, data progress, and state consistency.
- Health monitoring, logs, and coordination must scale without centralized bottlenecks.
- Slow nodes, degraded storage, and intermittent anomalies can consume resources for long periods without terminating processes.

### 5.3 Technical Competitiveness Goal

> Build a hierarchical fault-tolerant runtime for 100,000-device jobs, isolating failure domains, sustaining healthy portions, and efficiently recovering/rejoining failed portions. Preserve training quality and productive output under an agreed failure model while reducing impact, rollback loss, and long-term resource waste.

Local recovery units must respect DP replicas and TP/CP/EP/PP dependencies. Enable local recovery where complete state or redundancy exists; retain correct whole-job recovery elsewhere. Coordinate with cluster scheduling, replacement resources, networks, and storage through explicit interfaces.

### 5.4 Concrete Work

| ID | Work package | Work | Deliverables |
| --- | --- | --- | --- |
| 3.1 | Fault domains and hierarchical health monitoring | Define interruption, degradation, and correlated failures; derive failure/recovery units from parallel dependencies; collect hierarchical health/progress data and identify slow nodes, degraded links, and prolonged lack of progress. | Failure models, domain design, hierarchical monitoring, and fault localization. |
| 3.2 | Complete training-state protection | Save model, optimizer, scheduler, RNG, data position, and external state; optimize asynchronous/hierarchical checkpoints and fast transfers; handle interrupted saves and incomplete state. | Complete consistent protection, save/load baselines, storage-failure validation. |
| 3.3 | Reconfigurable communication and recovery | Support member departure, group adjustment, failed-replica recovery, and catch-up; preserve step, gradient aggregation, and data-progress consistency; reduce whole-job recovery and rejoining costs. | Fault-tolerant communication/recovery protocols, applicability, correctness, and latency reports. |
| 3.4 | Isolation and resource replacement | Coordinate with schedulers to isolate faulty nodes and obtain replacements; limit propagation; support topology changes and elastic resumption where semantics/state permit. | Scheduler/resource interfaces, replacement workflows, bounded elastic recovery. |
| 3.5 | Graded scale validation and fault exercises | Simulate the control plane at 100,000 ranks, inject faults on available hardware, and jointly validate target scale; cover repeated failures, stragglers, and communication/storage faults; measure retained progress and lost device-hours. | Graded validation, fault-injection tools, long-run and post-recovery quality reports. |

### 5.5 Acceptance Metrics and Scale Definitions

| Metric | Acceptance definition |
| --- | --- |
| Effective training output | Retained useful progress and resource cost within an observation window, also recording waits, rollback, and repeated computation. |
| Recovery time | Separately record detection, resumed progress on healthy portions, and rejoining of failed portions; report P95/P99. |
| Failure impact | Counts and fractions of paused, restarted, or isolated devices, groups, and replicas. |
| Progress loss and mechanism overhead | Rolled-back training, lost device-hours, checkpoint/monitoring overhead, and recovery-state transfer cost. |
| Recovery correctness | Data progress, parameters, and optimizer states meet agreed semantics; post-recovery numerics and convergence satisfy acceptance criteria. |

Deliver scale validation separately:

1. **100,000-device design and simulation**: Validate coordination protocols, control-plane scalability, and external-service pressure.
2. **Fault exercises on available hardware**: Validate real communication, state restoration, device execution, and failure propagation.
3. **Joint validation on 100,000 physical devices**: Validate final-scale capability with explicit models, duration, failure types, and frequencies.

State the limits of design, simulation, and smaller-scale results separately. Confirm 100,000-device training claims with corresponding hardware evidence.

## 6. Capability Four: A Multicore Parallel Development Framework—Execution and Operator Development Efficiency

### 6.1 Trends for 2027

**Public signals.** Dist-MoE provides fused communication/computation for training forward and backward. Ascend's 2026 Ascend C upgrade emphasizes hybrid SIMD/SIMT programming and tile-oriented PTO instructions, expanding opportunities to coordinate core-level computation, movement, and communication.
References: [Dist-MoE](https://github.com/meta-pytorch/dist_moe), [Ascend programming upgrades](https://www.huawei.com/en/news/2026/9/hc-agentic-thinkpro-pto-cann).

**Planning assessment.** New operators and fusion patterns make core-level work assignment and fine-grained scheduling increasingly central to training performance. Reusable frameworks are needed to reduce the cost of taking operators from algorithm design to training readiness and target performance.

### 6.2 Key Challenges

Matrix computation, vector processing, data movement, and communication have different execution characteristics. Overlapping them requires core assignment, tile partitioning, buffer lifetimes, and event dependencies.
Writing these system mechanisms separately for every operator raises development, validation, tuning, and cross-generation migration costs. Forward optimization must also be validated with backward, gradient accumulation, memory policies, and low-precision training.

### 6.3 Technical Competitiveness Goal

> Build a programmable, composable, tunable multicore parallel framework. Use task abstractions, automatic scheduling, and shared mechanisms to improve execution and development efficiency, enabling developers to reach correctness, performance targets, and training integration quickly.

The framework targets heterogeneous on-chip cores and connects to cross-device communication. Developers describe semantics, necessary kernels, and task dependencies; the framework handles common core mapping, orchestration, memory planning, and execution generation.

**Existing foundation.** Multicore already provides fused MoE forward/backward, task graphs, scheduling, and Torch integration. Work in 2027 expands reuse, completes tuning/validation tools, and adds representative scenarios based on measured benefit.
References: [Multicore Guide](../hyper_parallel/core/multicore/README.md), [Multicore Architecture](../hyper_parallel/core/multicore/docs/architecture.md).

### 6.4 Concrete Work

| ID | Work package | Work | Deliverables |
| --- | --- | --- | --- |
| 4.1 | Unified task representation and APIs | Describe compute, communication, movement, synchronization, and dependencies; support composable subgraphs and operator templates; define extensions and kernel integration. | Multicore task representation, development APIs, templates, and examples. |
| 4.2 | Automatic scheduling/resource planning | Optimize tiles, Cube/Vector allocation, task pipelines, and buffer reuse; generate execution/synchronization configuration; adapt to dynamic loads and target hardware generations. | Scheduling/planning implementation, execution generation, overhead/performance reports. |
| 4.3 | Shared mechanisms and training components | Reuse communication, synchronization, transfers, and pipelines; provide forward/backward, gradient accumulation, and low-precision templates; expose stable interfaces for model-level Codegen composition. | Reusable tasks/runtime components, training templates, Codegen examples. |
| 4.4 | Tuning and correctness tools | Provide parameter search, core-level profiling, and bottleneck analysis; diagnose synchronization/memory issues; improve numerical checks and benchmarks to shorten tuning time. | Tuning tools, diagnostic reports, correctness/performance workflows. |
| 4.5 | Operator development efficiency and scenarios | Start with MoE and extend hotspot operators/fused subgraphs according to benefit; record development, tuning, and training integration time against handwritten workflows; validate migration costs. | Representative implementations, efficiency benchmarks, migration reports, end-to-end gains. |

### 6.5 Operator Development Competitiveness and Acceptance Metrics

| Metric | Measurement and comparison |
| --- | --- |
| Time to first correct execution | Elapsed time and person-days from agreed operator semantics/I/O to passing required correctness checks. |
| Time to target performance | Implementation/tuning time, human effort, and evaluation resources under identical hardware, shape coverage, numerical and performance thresholds. |
| Training integration time | Time from operator implementation to forward/backward integration, parameter updates, and target workload validation. |
| Handwritten code and mechanism reuse | Manually maintained code volume and reuse of agreed communication, synchronization, memory, and scheduling mechanisms; count generated code separately. |
| Cross-generation migration cost | Changes, tuning time, and validation cost under identical semantics and acceptance criteria. |
| Execution/training benefits | Core utilization, exposed communication time, transfer overhead, and full-workload throughput/device-hour improvements. |

Compare against existing handwritten workflows under identical numerical and performance requirements for pre-agreed representative operators and fused subgraphs. Accept development efficiency and execution performance together rather than judging competitiveness solely by supported operator count.

## 7. Shared Quality and Diagnostic Requirements

| Requirement | Work | Main acceptance evidence |
| --- | --- | --- |
| Training semantics/state contracts | Define loss normalization, variable-length weighting, gradient accumulation, global clipping, shared parameters, and external state; standardize generation, recomputation, and recovery. | Executable semantic checks, boundary cases, consistency reports. |
| Layered correctness | Validate from independent references to generated/optimized implementations, covering operators, modules, one-step updates, short-run convergence, and long-run quality. | Independent references and layered numerical/convergence reports. |
| Optimization combinations | Cover committed combinations of parallelism, compilation, recomputation, offload, and low precision; validate risky interactions and limits. | Coverage matrix, quality deviations, known limitations, release gates. |
| Distributed diagnosis/reproduction | Correlate compute, communication, inputs, gradients, and state events across ranks; preserve configuration, environment, and data identity. | First-anomaly localization, reproducibility, regression detection/localization time. |
| Silent error detection | Inject finite-value perturbations, missing state, or incorrect redistribution in test environments; distinguish normal numerical variation from faults. | Detection/false-positive rates, automated regression, version quality reports. |

Tie metrics to specific models, hardware, workloads, and training durations. Local numerical success cannot replace full training-quality acceptance.

## 8. Proposed Quarterly Schedule

| Phase | Codegen/personalized frameworks | Complex-model parallelism | 100,000-device high availability | Multicore/operator efficiency |
| --- | --- | --- | --- | --- |
| Q1 | Define generation contracts, agent APIs, extension tasks, and efficiency baselines. | Build hardware/workload profiles and select representative workloads. | Define failure models, domains, scale-validation paths, and external interfaces. | Define task/extension APIs and handwritten baselines for representative operators. |
| Q2 | Deliver initial personalized artifacts, generation/validation loops, and extension points. | Complete priority MoE/CP paths, joint memory policies, and performance baselines. | Complete full-state protection, fast whole-job recovery, and initial local recovery checks. | Deliver shared components, training templates, and initial operator development tools. |
| Q3 | Expand agent optimization and validate customization preservation/upgrades. | Extend all-modal/world-model scenarios and test scale/extreme workloads. | Expand reconfigurable communication, catch-up, and fault exercises; simulate control-plane scale. | Expand subgraphs/hardware and validate autotuning and migration efficiency. |
| Q4 | Accept generation coverage, efficiency, quality, and performance; finalize deliverable scope. | Accept capability limits, scaling efficiency, and end-to-end costs. | Complete target-scale joint long-run/recovery acceptance according to resources, stating measured scale limits. | Compare development cycles, reuse, and training gains against baselines. |

Numerical/convergence validation, diagnostic tools, and performance regression span every quarter. This schedule is a proposal.

## 9. Stage Gates and Project Initiation Requirements

### 9.1 Gates for Exploratory Capabilities

Expand agent optimization, dynamic parallel scheduling, local recovery, elastic resumption, and cross-scenario multicore scheduling in this order:

1. Validate semantics, state, and numerical correctness for bounded models, hardware, and workloads.
2. Measure overhead, development cost, and net end-to-end benefit; establish applicability.
3. After agreed conditions are met, expand models, hardware, optimization combinations, and cluster scale.

### 9.2 Inputs Required at Initiation

- Target users, models, training tasks, data, and workload distributions.
- GPU/NPU platforms, interconnects, cluster scale, and storage conditions.
- Codegen coverage matrix, personalized framework scope, agent tasks, and extension acceptance tasks.
- Representative world-model architectures, training paths, modalities, and effective-work definitions.
- Operator/fused-subgraph inventory, performance thresholds, and handwritten baselines.
- Resources for 100,000-device design, simulation, and hardware validation; failure frequency, continuous-run duration, and allowed progress loss.
- Work-package owners, staffing, cross-team dependencies, quantitative targets, and quarterly delivery boundaries.

Set numerical targets from this scope and these baselines; do not generalize gains from one model or experiment to every scenario.
