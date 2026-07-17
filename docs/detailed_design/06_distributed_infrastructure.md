# Hyper-Parallel 分布式基础设施详细设计

> 参考实现：AutoModel `components/distributed/config.py`、`mesh.py`、`fsdp2.py`、`parallelizer.py`
> 上下文设计：01_hf_compatibility_layer.md、05_dual_mode_dtensor_parallel_strategy.md

---

## 1. 模块职责

定义分布式训练的**拓扑构建**（MeshContext）、**策略配置**（FSDP2Config / DistributedSetup）、**DP 维度包裹**（FSDP2Manager）以及**生产模式零拷贝**（_local_params_context）。

### 核心文件

| 文件 | 职责 |
|------|------|
| `components/distributed/config.py` | `DistributedSetup` — 拓扑 + 策略配置的统一容器 |
| `components/distributed/mesh.py` | `MeshContext` / `ParallelismSizes` / `MeshAxisName` — DeviceMesh 构建与拓扑查询 |
| `components/distributed/fsdp2.py` | `FSDP2Manager` — FSDP2 包裹（DP 维度） |
| `components/distributed/parallelizer.py` | `fsdp2_strategy_parallelize()` — FSDP2 + TP 联合分片 |
| `components/distributed/dtensor_utils.py` | `_local_params_context` re-export 入口（canonical 实现在 `sharding/apply.py`，见 05 §4.4.1） |

### 涉及删除的旧代码

| 旧代码 | 替代方案 |
|--------|---------|
| 分散在 trainer 中的 `init_process_group` 调用 | `initialize_distributed` + `DistributedSetup.build()` |
| 旧的 `TPManager` / `DDPManager` 类 | `FSDP2Manager`（统一 DP 维度） |

### 1.1 独立使用

本模块零依赖 `recipes/`、`_transformers/`、`components/models/`。可脱离 Hyper-Parallel 训练流程独立使用：

```python
from hyper_parallel.components.distributed import (
    MeshContext, ParallelismSizes, FSDP2Manager
)
from hyper_parallel.components.distributed.config import DistributedSetup

# 独立构建分布式拓扑
# P2: 正确用法——通过 build() 类方法创建，传入 strategy 和 parallelism_sizes
setup = DistributedSetup.build(strategy="fsdp2", parallelism_sizes=ParallelismSizes(tp_size=4))
mesh = setup.mesh_context
# → mesh.tp_size=4, mesh.device_mesh 可直接用于任何 DTensor 操作

# 等价的更完整示例（tp=4, cp=2）：
setup = DistributedSetup.build(
    strategy="fsdp2",
    parallelism_sizes=ParallelismSizes(tp_size=4, cp_size=2),
)
# → mesh.device_mesh = DeviceMesh("cuda", (dp, 2, 4), ("dp_shard_cp", "cp", "tp"))
```

### 1.2 与现有代码的关系与迁移路径

本模块 `components/distributed/` 是一套全新设计，但与现有 `hyper_parallel/core/` 代码是
**封装复用**关系，而非推倒重写：

| 现有模块 | 关系 | 说明 |
|---------|------|------|
| `core/dtensor/` | **复用** | DTensor ops（25k+ 行），作为 DTensor dispatch 的底层实现保留。新设计中的 `distribute_tensor()`、`DTensor.redistribute()` 等操作继续走 `core/dtensor/` 的实现路径 |
| `core/shard/` | **复用** | `core/shard/` 底层算子保留为 DTensor dispatch 后端（被 `DTensor.redistribute()` 和 `distribute_tensor()` 内部复用）；旧的高层 shard plan 编排 API 被 `ShardingPlanner`/`ShardingApplier` 替代。用户侧代码只能通过新的 `components/distributed/` 入口使用分片功能。 |
| `core/fully_shard/` (HSDP) | **复用并扩展** | `_build_layout_driven_group_info()`、`all_reduce_grad()`、`DTENSOR_UNIFIED` 模式——新设计中的 FSDP2Manager 复用此梯度同步机制。新增 `tp_grad_info` 旁路用于 TP 维度梯度 all-reduce |
| `core/context_parallel/` | **复用** | CP ring attention 实现保留。新设计中由 `_wrap_cp_inner_attention` 在编译期注入 CP 通信，CP 的 ring attention 核心逻辑仍走 `core/context_parallel/` |
| `dmodule/` | **替换** | 旧的运行时 redistribute 逻辑 → 新 `PrecompiledBoundary` 编译期化。`dmodule/` 中的 DTensor 自动传播机制在生产模式下不再使用 |
| `models/*/parallelize.py` | **替换** | 过程式硬编码分片 → `ARCH_OVERRIDES` + `ShardingPlanner` 声明式自动推导（见 05 文档 §4） |

**关键集成点**：

```python
# _local_params_context 在 build 期一次性解包 DTensor → plain tensor
# 解包后的参数由 FSDP/HSDP 以 LOCAL_PARAM + tp_grad_info 管理
# TP 梯度 all-reduce 复用现有 HSDP layout-driven 梯度同步
# 详见 05 文档 §6.7
```

**迁移顺序**：先基于现有 `core/dtensor` 和 `core/fully_shard` 搭建新 `components/distributed/`，
待 ShardingPlanner + PrecompiledBoundary 验证通过后，逐步替换 `models/*/parallelize.py` 和 `dmodule/`。

---

## 2. 总入口调用时序

分布式基础设施在 `main()` 的**最早阶段**初始化——在任何模型/数据组件构建之前。

```
main()
├─① cfg = load_yaml_config("train.yaml")                             # 01_hf_compatibility_layer.md §2
├─② cfg = RecipeConfig(cfg)                                          # 01 §3
│
├─③ recipe = FinetuneRecipe()
├─④ recipe.setup(cfg)
│   │
│   ├─④.1 self.dist_env = initialize_distributed("nccl")             # torch.distributed.init_process_group
│   │
│   ├─④.3 self.distributed_setup = create_distributed_setup_from_config(cfg)  # ★ 拓扑构建
│   │   │
│   │   ├─④.3.1 ParallelismSizes(tp_size=4, cp_size=2, pp_size=1, ...)
│   │   │                                                             # §3.1: 从 YAML 提取并行度
│   │   │
│   │   ├─④.3.2 MeshContext.build(strategy_config, parallelism_sizes) # §3.2: 构建 DeviceMesh
│   │   │   ├─ _validate_parallelism_sizes(tp, cp, pp, ep, dp, world_size)
│   │   │   │   # heads % tp == 0, world_size == tp * cp * pp * ep * dp, ep ≤ dp ...
│   │   │   ├─ DeviceMesh("cuda", mesh_tensor, ("cp", "tp"))         # TP+CP mesh
│   │   │   ├─ DeviceMesh("cuda", mesh_tensor, ("dp_shard_cp", ...)) # DP+CP mesh
│   │   │   ├─ DeviceMesh("cuda", mesh_tensor, ("ep_shard", ...))    # EP mesh (MoE)
│   │   │   └─ 验证 axis names ∈ MeshAxisName
│   │   │
│   │   └─④.3.3 DistributedSetup(mesh_context, strategy_config, ...) # §3.3: 统一配置容器
│   │
│   ├─④.4 self.mesh = self.distributed_setup.mesh_context            # §3.2: 所有组件通过 mesh 查询拓扑
│   │   ├─ mesh.tp_size → 4     (@property 从 device_mesh["tp"] 读取)
│   │   ├─ mesh.cp_size → 2
│   │   ├─ mesh.dp_size → N     (从 device_mesh["dp_shard_cp"] 读取)
│   │   ├─ mesh.tp_rank → 0..3
│   │   └─ mesh.device_mesh → DeviceMesh("cuda", (2,4), ("cp","tp"))
│   │
│   └─④.3.2 instantiate_infrastructure(distributed_setup, device)    # 01 §8
│       ├─ ShardingPlanner()                                          # 05
│       ├─ FSDP2Manager(strategy_config, mesh)  if strategy          # ★ §4 (收 MeshContext)
│       │   └─ 详见 §4
│       └─ AutoPipeline(pipeline_config, mesh)          if pp > 1
│
├─ ... (后续: _build_model → ShardingPlanner → FSDP2.parallelize)

── FSDP2 应用（canonical meta 链路，在 _build_model 中）──

_build_model()                                                        # 01 §6.3
├─④.4.5.7 sharding_planner.plan() → ShardingPlan                     # 05 §4: TP/CP/SP 分片
├─④.4.5.8 apply_sharding_plan(model, plan, mesh.device_mesh)         # 05 §4: DTensor 应用
│   └─ Phase C 含 _local_params_context 解包；内部调 build_tp_grad_info(plan, tp_mesh)
│       → 返回 (model, tp_grad_info)（build_tp_grad_info 是内部调用，非独立步骤）
│       详见 §5
├─④.4.5.10 fsdp2_manager.parallelize(model, tp_grad_info=tp_grad_info)  # ★ §4.2: meta 上 fully_shard（唯一一次）
│   └─ fsdp2_strategy_parallelize(model, dp_mesh=..., tp_grad_info=...)  # §4.3
│       ├─ fully_shard(block, mesh=dp_mesh, mp_policy=..., reshard_after_forward=True, tp_grad_info=...)
│       ├─ 应用 TP shard plan（已由 ShardingPlanner 产生）
│       ├─ 应用 activation checkpointing
│       └─ 可选: async TP, per-layer compile, prefetch
├─ to_empty(device=...) → 物化 sharded 参数                           # FSDP2 canonical 顺序
└─④.4.5.11 load_base_model(model, device, path, adapter=..., mesh=mesh.device_mesh)  # 04 §5: 权重写入 sharded 参数
```

**与 01、05 文档的时序衔接**：

```
main()
├─④.1 initialize_distributed         # 本文档 §3
├─④.3 DistributedSetup.build()       # 本文档 §3
│   └─ MeshContext.build()           # 本文档 §3.2
├─④.3.2 FSDP2Manager()               # 本文档 §4 (创建)
└─_build_model()
    ├─④.4.5.7 sharding_planner.plan()  # 05 §4
    ├─④.4.5.8 apply_sharding_plan()    # 05 §4
    │   └─ _local_params_context       # 本文档 §5
    └─④.4.5.10 fsdp2.parallelize()     # 本文档 §4.2 (应用)
```

---

## 3. DistributedSetup + MeshContext：拓扑构建

### 3.1 ParallelismSizes：用户意图

```python
# components/distributed/mesh.py
#
# P2: 以下类型别名的实际定义位于 config.py，此处仅声明引用关系：
# - DistributedStrategyConfig = FSDP2Config | DDPConfig | MegatronFSDPConfig
# - FSDP2Config: 见 §4.1，定义在 components/distributed/fsdp2.py
# - DDPConfig: 纯 DDP 配置（无 TP/CP 维度），定义在 components/distributed/config.py
# - MegatronFSDPConfig: Megatron 风格 FSDP 配置，定义在 components/distributed/config.py

@dataclass(frozen=True, kw_only=True)
class ParallelismSizes:
    """用户声明的并行度——这是"期望"，不是运行时拓扑。"""
    dp_size: int | None = None       # None → 自动推导
    dp_replicate_size: int | None = None  # FSDP2 HYBRID_SHARD
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1                 # Expert Parallel (MoE)
```

### 3.2 MeshContext：运行时拓扑

```python
# components/distributed/mesh.py
#
# 依赖符号（以下符号由对应模块提供，本文件顶部应 import）：
# - DistributedStrategyConfig / FSDP2Config / DDPConfig / MegatronFSDPConfig:
#     from .config import (DistributedStrategyConfig, FSDP2Config,
#                          DDPConfig, MegatronFSDPConfig)
# - _resolve_strategy_config(strategy: str | DistributedStrategyConfig):
#     定义在 config.py，将 str ("fsdp2"/"ddp"/"megatron_fsdp") 映射到对应 Config 类。
# - get_world_size_safe() -> int:
#     定义在 mesh.py（或 utils.py），封装 dist.get_world_size()，
#     未初始化时返回 1（便于单进程测试）。
# - PipelineConfig / MoEParallelizerConfig: 定义在各自模块，由 config.py re-export。
# - MixedPrecisionPolicy / CPUOffloadPolicy / ShardingPlan / Placement /
#   DeviceMesh / DTensor: 由 torch.distributed.fsdp2 / torch.distributed.tensor /
#   本仓库 components.distributed.sharding_plan 提供。
# P2: 实际 import 在实现时补全（当前为注释依赖声明，代码实现时替换为实际 import 语句）

class MeshAxisName(str, Enum):
    """DeviceMesh 的轴名——所有 mesh 维度使用规范名称。"""
    PP = "pp"
    DP = "dp"
    DP_REPLICATE = "dp_replicate"
    DP_SHARD = "dp_shard"
    DP_SHARD_CP = "dp_shard_cp"
    DP_CP = "dp_cp"             # P2: 保留位——当前所有 mesh 构建路径均不使用此轴名，
                                # 预留给未来 DP+CP 联合优化场景（如 DP/CP fused all-gather）。
    CP = "cp"
    TP = "tp"
    EP = "ep"
    EP_SHARD = "ep_shard"


@dataclass
class MeshContext:
    """运行时分布式拓扑上下文。

    所有 size 和 rank 属性都是 @property，从实际 DeviceMesh 读取：
    - tp_size → device_mesh["tp"].size()
    - tp_rank → device_mesh.get_local_rank("tp")
    - dp_size → device_mesh["dp_shard_cp"].size()（HYBRID_SHARD 时为
      dp_shard.size() * dp_replicate.size()，但 EP 从 DP 切出后应满足
      dp_size % ep_size == 0）
    """

    # 实际的 DeviceMesh 对象
    device_mesh: DeviceMesh           # 主 mesh: ("dp_shard_cp", "cp", "tp")
    device: str = "cuda"              # mesh 所在 device（offload 链路使用）
    moe_mesh: DeviceMesh | None = None  # MoE mesh: ("ep_shard",) 从 DP 切出

    @property
    def tp_size(self) -> int:
        return self.device_mesh["tp"].size() if "tp" in self.device_mesh.mesh_dim_names else 1

    @property
    def cp_size(self) -> int:
        return self.device_mesh["cp"].size() if "cp" in self.device_mesh.mesh_dim_names else 1

    @property
    def pp_size(self) -> int:
        return self.device_mesh["pp"].size() if "pp" in self.device_mesh.mesh_dim_names else 1

    @property
    def dp_size(self) -> int:
        # canonical：从 mesh 轴读取，而非 world_size 除法
        # - 普通 FSDP2：轴序 ("dp_shard_cp", "cp", "tp")，dp_shard_cp.size() 即 dp_size
        # - HYBRID_SHARD：轴序 ("dp_replicate", "dp_shard_cp", "cp", "tp")，
        #   dp_size = dp_replicate.size() * dp_shard_cp.size()
        # - EP 从 DP 切出，dp_size 必须能被 ep_size 整除（由 _validate_parallelism_sizes 保证）
        names = self.device_mesh.mesh_dim_names
        if "dp_replicate" in names and "dp_shard_cp" in names:
            return self.device_mesh["dp_replicate"].size() * self.device_mesh["dp_shard_cp"].size()
        if "dp_shard_cp" in names:
            return self.device_mesh["dp_shard_cp"].size()
        if "dp_replicate" in names and "dp_shard" in names:
            return self.device_mesh["dp_replicate"].size() * self.device_mesh["dp_shard"].size()
        if "dp_shard" in names:
            return self.device_mesh["dp_shard"].size()
        if "dp" in names:
            return self.device_mesh["dp"].size()
        # P2 修复：fallback 必须除 ep_size（dp 此处不含 ep，EP 从 dp 切出）
        ep = self.ep_size
        dp_fallback = get_world_size_safe() // (self.tp_size * self.cp_size * self.pp_size * ep)
        # P2: fallback 计算路径未经 DeviceMesh 轴存在性验证——仅在
        # 无法从 mesh 轴直接读取时才走到此分支（如测试 mesh、旧代码兼容 mesh）。
        # 实现时加 assert 检查 dp_fallback >= 1，避免除零/负值泄漏：
        assert dp_fallback >= 1, \
            f"dp_size fallback 异常: {dp_fallback} (world={get_world_size_safe()}, " \
            f"tp={self.tp_size}, cp={self.cp_size}, pp={self.pp_size}, ep={ep})"
        return dp_fallback

    @property
    def tp_rank(self) -> int:
        return self.device_mesh.get_local_rank("tp") if "tp" in self.device_mesh.mesh_dim_names else 0

    @property
    def dp_rank(self) -> int:
        # 第六轮 P1 修复：供 03 BaseRecipe._get_dp_rank 调用。
        # 优先按 DP 轴取 local_rank；轴名随策略而异（dp_shard_cp / dp_shard / dp / dp_replicate）。
        names = self.device_mesh.mesh_dim_names
        for ax in ("dp_shard_cp", "dp_shard", "dp", "dp_replicate"):
            if ax in names:
                return self.device_mesh.get_local_rank(ax)
        return 0

    @property
    def pp_rank(self) -> int:
        # 第六轮 P1 修复：供 03 BaseRecipe._get_pp_rank 调用。
        return self.device_mesh.get_local_rank("pp") if "pp" in self.device_mesh.mesh_dim_names else 0

    @property
    def tp_mesh(self) -> "DeviceMesh | None":
        # N2 修复：供 build_tp_grad_info(plan, mesh.tp_mesh) 调用
        # P1 修复：无 tp 轴时返回 None（与 05 _get_tp_submesh 返回 None 一致），
        # 避免 KeyError。
        names = self.device_mesh.mesh_dim_names
        return self.device_mesh["tp"] if "tp" in names else None

    @property
    def cp_mesh(self) -> "DeviceMesh | None":
        # P1 修复：无 cp 轴时返回 None（不再 fallback 到 tp_mesh——
        # cp_mesh 与 tp_mesh 语义不同，fallback 会掩盖配置错误），避免 KeyError。
        names = self.device_mesh.mesh_dim_names
        return self.device_mesh["cp"] if "cp" in names else None

    @property
    def dp_mesh(self) -> "DeviceMesh | None":
        # 第六轮 P1 修复：供 03 `self.mesh.dp_mesh` 调用（_dp_all_reduce_avg 等）。
        # 优先取 "dp_shard_cp" 子 mesh（FSDP2/HYBRID_SHARD），退化为 "dp" / "dp_replicate"。
        names = self.device_mesh.mesh_dim_names
        if "dp_shard_cp" in names:
            return self.device_mesh["dp_shard_cp"]
        if "dp" in names:
            return self.device_mesh["dp"]
        if "dp_replicate" in names:
            return self.device_mesh["dp_replicate"]
        return None

    @property
    def ep_size(self) -> int:
        # P2 修复：dp_size fallback 与 EP 拓扑查询共用
        names = self.device_mesh.mesh_dim_names
        return self.device_mesh["ep_shard"].size() if "ep_shard" in names else 1

    @classmethod
    def build(
        cls,
        strategy_config: DistributedStrategyConfig,
        parallelism_sizes: ParallelismSizes,
        world_size: int | None = None,
    ) -> "MeshContext":
        """从并行度配置构建实际的 DeviceMesh 对象。

        构建逻辑（按策略类型）：
        - DDP: 无 tp/cp → dp mesh only
        - FSDP2: tp + cp → ("dp_shard_cp", "tp") 或 ("dp_shard", "tp")
        """
        if world_size is None:
            world_size = get_world_size_safe()

        tp = parallelism_sizes.tp_size
        cp = parallelism_sizes.cp_size
        pp = parallelism_sizes.pp_size
        ep = parallelism_sizes.ep_size
        # 读取用户声明的 dp_size / dp_replicate_size（此前两个字段在 build 中被忽略）
        dp_replicate = parallelism_sizes.dp_replicate_size or 1

        # dp 自动推导（world_size 必须能被 tp*cp*pp*ep 整除）；
        # 若用户显式声明 dp_size 则优先采用并校验。
        # 注意：此处的 dp **不含** ep（EP 从 dp 切出后 per_ep = dp // ep），
        # 故 world_size = tp*cp*pp*ep*dp（dp 排除 ep）。
        dp = parallelism_sizes.dp_size or (world_size // (tp * cp * pp * ep))

        # 校验（arity 与定义一致：6 参）
        # 注意：_validate 校验的是整体拓扑一致性（含 pp），与 mesh 构造策略解耦——
        # FSDP2 mesh 不把 pp 计入轴（见 _build_fsdp2_mesh），但 pp 仍需参与整体
        # world_size 一致性校验。strategy=fsdp2 且 pp>1 是允许的：AutoPipeline
        # 在 FSDP2 mesh 之上独立管理 PP stage 切分（pp 不进 fsdp2 mesh）。
        _validate_parallelism_sizes(tp, cp, pp, ep, dp, world_size)

        # 构建 mesh tensor
        # 例: tp=4, cp=2, ep=1, world_size=64 → dp=8
        # mesh_tensor = reshape(arange(64), (8, 2, 4))
        # device_mesh = DeviceMesh("cuda", mesh_tensor, ("dp_shard_cp", "cp", "tp"))

        if isinstance(strategy_config, FSDP2Config):
            ctx = cls._build_fsdp2_mesh(
                tp, cp, pp, world_size,
                dp_replicate_size=dp_replicate, ep_size=ep,
            )
        elif isinstance(strategy_config, MegatronFSDPConfig):
            ctx = cls._build_megatron_mesh(tp, cp, pp, world_size)
        else:  # DDPConfig
            ctx = cls._build_ddp_mesh(world_size)

        # MoE mesh：EP 从 DP 切出，在 build() 中赋值（此前从不赋值）
        # P2 修复：ep>1 但 mesh 无 ep_shard 轴时（Megatron/DDP builder 不产 ep_shard）
        # 显式报错，而非静默返回 moe_mesh=None。
        if ep > 1:
            names = ctx.device_mesh.mesh_dim_names
            if names and "ep_shard" in names:
                ctx.moe_mesh = ctx.device_mesh["ep_shard"]
            else:
                raise ValueError(
                    f"ep_size={ep} > 1 但 strategy={type(strategy_config).__name__} "
                    f"的 mesh 不含 ep_shard 轴（mesh_dim_names={names}）。"
                    f"EP 目前仅在 FSDP2Config 策略下支持；Megatron/DDP 不产 ep_shard。"
                )
        return ctx

    # ── mesh 构建器（此前为黑盒，此处补全） ──

    @classmethod
    def _build_fsdp2_mesh(
        cls, tp, cp, pp, world_size, *,
        dp_replicate_size: int = 1, ep_size: int = 1,
    ) -> "MeshContext":
        """构建 FSDP2 canonical mesh。

        - 无 HYBRID_SHARD（dp_replicate_size==1）：轴序 ("dp_shard_cp", "cp", "tp")
          其中 dp_shard_cp 折叠了 dp_shard × cp（dp 维度的所有 shard rank）
        - HYBRID_SHARD（dp_replicate_size>1）：拆 DP_REPLICATE / DP_SHARD 两轴，
          轴序 ("dp_replicate", "dp_shard_cp", "cp", "tp")
        - EP>1：EP 从 DP 切出，在 dp_shard 维内再切出 ep_shard 子轴，
          轴序 ("dp_replicate"?,"dp_shard_cp","cp","tp","ep_shard")
          （ep_shard 是 dp_shard 的细分，保证 ep_size 整除 dp_size）

        **PP 与 FSDP2 mesh 的关系**（第六轮 P0 修复）：PP 维度由 AutoPipeline 独立管理，
        FSDP2 mesh 只覆盖 DP/CP/TP/EP。若把 pp 计入 `dp_total`（`dp_total = world_size//(tp*cp)`
        在 pp>1 时 = dp*pp*ep），pp 因子会偷偷塞进 `dp_shard_cp` 轴，导致 `dp_size` property
        返回 dp*pp、`fully_shard` 在 PP group 间 all-gather 破坏 PP stage 隔离。因此
        **显式禁止 pp>1 + FSDP2**：`assert pp == 1`。PP 与 FSDP2 联用（per-stage 子 mesh）
        为 future work，由 AutoPipeline 单独管 mesh。
        """
        # P0 修复：显式禁止 pp>1 + FSDP2，避免 pp 污染 dp_shard_cp 轴。
        assert pp == 1, (
            "pp>1 + FSDP2 暂不支持：FSDP2 mesh 只覆盖 DP/CP/TP/EP，pp 会污染 dp_shard_cp 轴"
            "（dp_size 返回 dp*pp、fully_shard 跨 PP stage all-gather）。"
            "PP 由 AutoPipeline 独立管 mesh，per-stage FSDP2 子 mesh 为 future work。"
        )
        # dp_total 含 EP（= dp*ep，与 build() 中排除 ep 的 dp 不同）；pp 已断言为 1。
        dp_total = world_size // (tp * cp)
        dp_shard = dp_total // dp_replicate_size           # 节点内 shard（含 ep）
        assert dp_shard % ep_size == 0, "ep_size 必须整除 dp_shard"

        device = "cuda"
        if dp_replicate_size == 1 and ep_size == 1:
            mesh_tensor = torch.arange(world_size).reshape(dp_total, cp, tp)
            dim_names = ("dp_shard_cp", "cp", "tp")
        elif dp_replicate_size > 1 and ep_size == 1:
            mesh_tensor = torch.arange(world_size).reshape(
                dp_replicate_size, dp_shard, cp, tp)
            dim_names = ("dp_replicate", "dp_shard_cp", "cp", "tp")
        else:  # ep_size > 1：从 dp_shard 切出 ep_shard
            per_ep = dp_shard // ep_size
            mesh_tensor = torch.arange(world_size).reshape(
                dp_replicate_size, ep_size, per_ep, cp, tp)
            # dim_names 顺序与 reshape 逐维对齐：轴0=dp_replicate, 轴1=ep_shard,
            # 轴2=dp_shard_cp(per_ep), 轴3=cp, 轴4=tp
            # P2: 当 dp_replicate_size==1 时，mesh 多一个大小为 1 的 dp_replicate 维度
            # （reshape 第一个维度=1）。该维度不参与通信但存在于 mesh 轴列表中，
            # 后续可通过 squeeze 去掉或保留（保留不影响正确性，仅多一个冗余轴）。
            dim_names = ("dp_replicate", "ep_shard", "dp_shard_cp", "cp", "tp")
        device_mesh = DeviceMesh(device, mesh_tensor, dim_names)
        # 自检：dim_names[i] 必须对应 reshape 第 i 维语义（N1 修复）
        assert tuple(device_mesh[n].size() for n in dim_names) == tuple(mesh_tensor.shape), \
            f"EP mesh dim_names 与 reshape 轴错位: {dim_names} vs shape {tuple(mesh_tensor.shape)}"
        # P2: device_mesh[n].size() 的 key 访问方式较脆弱——若 dim_names 包含 mesh 中
        # 不存在的轴名会直接 KeyError。实现时增加 try/except 或显式校验 names 子集。
        return cls(device_mesh=device_mesh, device=device)

    @classmethod
    def _build_megatron_mesh(cls, tp, cp, pp, world_size) -> "MeshContext":
        """Megatron-style mesh：("dp", "pp", "tp", "cp")。签名与 _build_fsdp2_mesh 对齐。"""
        dp = world_size // (tp * cp * pp)
        mesh_tensor = torch.arange(world_size).reshape(dp, pp, tp, cp)
        device_mesh = DeviceMesh("cuda", mesh_tensor, ("dp", "pp", "tp", "cp"))
        return cls(device_mesh=device_mesh, device="cuda")

    @classmethod
    def _build_ddp_mesh(cls, world_size) -> "MeshContext":
        """纯 DDP：单轴 ("dp",)。"""
        mesh_tensor = torch.arange(world_size).reshape(world_size)
        device_mesh = DeviceMesh("cuda", mesh_tensor, ("dp",))
        return cls(device_mesh=device_mesh, device="cuda")
```

### 3.3 DistributedSetup：统一配置容器

```python
# components/distributed/config.py

@dataclass(frozen=True)
class DistributedSetup:
    """解析后的分布式拓扑和执行策略。

    这是 DistributedSetup 的完整解析结果，由 Recipe 在 setup() 最早阶段创建，
    所有后续组件通过它查询拓扑信息。
    """
    mesh_context: MeshContext                        # 拓扑查询入口
    strategy_config: DistributedStrategyConfig | None = None  # FSDP2Config / DDPConfig
    pipeline_config: PipelineConfig | None = None             # PP 配置
    moe_parallel_config: MoEParallelizerConfig | None = None  # MoE EP 配置
    activation_checkpointing: bool | str = False              # 激活检查点

    @classmethod
    def build(
        cls,
        strategy: str | DistributedStrategyConfig = "fsdp2",
        parallelism_sizes: ParallelismSizes | None = None,
        pipeline_config: PipelineConfig | dict | None = None,
        activation_checkpointing: bool | str = False,
        **kwargs,
    ) -> "DistributedSetup":
        """从 YAML 配置构建完整的分布式 setup。

        这是 create_distributed_setup_from_config() 的实际实现。
        """
        strategy_config = _resolve_strategy_config(strategy)
        if parallelism_sizes is None:
            parallelism_sizes = ParallelismSizes()

        mesh_context = MeshContext.build(strategy_config, parallelism_sizes)
        return cls(
            mesh_context=mesh_context,
            strategy_config=strategy_config,
            pipeline_config=pipeline_config,
            activation_checkpointing=activation_checkpointing,
        )
```

**P2: `_resolve_strategy_config` stub 定义**——上述 `DistributedSetup.build()` 调用
`_resolve_strategy_config(strategy)`，该函数定义在 `config.py`，将字符串映射到对应
Config 类。实现时 stub 如下（由 §3.2 注释的 import 提供）：

```python
# components/distributed/config.py

def _resolve_strategy_config(strategy: str | DistributedStrategyConfig) -> DistributedStrategyConfig:
    """将 strategy 字符串映射到对应 Config 类。
    
    - "fsdp2" → FSDP2Config()
    - "ddp"   → DDPConfig()
    - "megatron_fsdp" → MegatronFSDPConfig()
    - 已为 Config 实例则直接返回。
    """
    if isinstance(strategy, DistributedStrategyConfig):
        return strategy
    if strategy == "fsdp2":
        return FSDP2Config()
    if strategy == "ddp":
        return DDPConfig()
    if strategy == "megatron_fsdp":
        return MegatronFSDPConfig()
    raise ValueError(f"Unknown strategy: {strategy}")
```

### 3.4 完整 Mesh 构建示例

以 `tp=4, cp=2, world_size=64` 为例：

```python
# YAML
distributed:
  tp: 4
  cp: 2
  strategy: fsdp2

# Step 1: 提取 sizes
sizes = ParallelismSizes(tp_size=4, cp_size=2)

# Step 2: dp 自动推导
# dp = 64 / (4 * 2 * 1) = 8

# Step 3: 构建 mesh_tensor
mesh_tensor = torch.arange(64).reshape(8, 2, 4)

# Step 4: 创建 DeviceMesh
device_mesh = DeviceMesh("cuda", mesh_tensor, mesh_dim_names=("dp_shard_cp", "cp", "tp"))

# Step 5: 创建 MeshContext
mesh = MeshContext(device_mesh=device_mesh)

# 查询:
mesh.tp_size    # → 4  (device_mesh["tp"].size())
mesh.cp_size    # → 2  (device_mesh["cp"].size())
mesh.dp_size    # → 8  (device_mesh["dp_shard_cp"].size())
mesh.tp_rank    # → 0..3  (device_mesh.get_local_rank("tp"))
```

---

## 4. FSDP2Manager：DP 维度包裹

### 4.1 FSDP2Manager 初始化

```python
# components/distributed/fsdp2.py

@dataclass
class FSDP2Config:
    """FSDP2 策略配置。"""
    sequence_parallel: bool = False
    activation_checkpointing: bool | str = False
    mp_policy: MixedPrecisionPolicy | None = None        # 混合精度策略
    offload_policy: CPUOffloadPolicy | None = None       # CPU offload
    reshard_after_forward: bool = True                    # FSDP reshard
    defer_fsdp_grad_sync: bool = True                    # 延迟梯度同步
    enable_async_tensor_parallel: bool = False            # 异步 TP all-reduce
    enable_compile: bool = False                          # per-layer compile
    enable_fsdp2_prefetch: bool = False                   # FSDP2 all-gather prefetch
    # prefetch 不需要 fork 新增 depth API——自研 fully_shard
    # （core/fully_shard/api.py）已提供
    # set_modules_to_forward_prefetch(modules) /
    # set_modules_to_backward_prefetch(modules)，传入按执行顺序排序的
    # FSDP block 列表即可实现多级 prefetch。depth 字段映射为"取有序
    # block 列表的前 depth 层注册 prefetch"，depth<=0 等价不开启。
    backward_prefetch_depth: int = 1                      # backward 参数 all-gather prefetch 深度
    forward_prefetch_depth: int = 1                       # forward 参数 all-gather prefetch 深度


class FSDP2Manager:
    """FSDP2 包裹器 —— 在 DTensor 分片完成后，对 DP 维度应用 FSDP2。

    生命周期：
    1. instantiate_infrastructure() 中创建（不应用）
    2. _build_model() 末尾调用 fsdp2_manager.parallelize(model)（应用）
    """

    def __init__(self, config: FSDP2Config, mesh: MeshContext):
        # canonical 2 参：收 MeshContext（携带 device_mesh + device），
        # 与 01 §8.1 `_instantiate_fsdp2(*, config, mesh)` 对齐。
        # device 由 MeshContext 携带——instantiate_infrastructure 传入的 device
        # 不再单独接收，避免双源（C6）。
        # P2: 运行时需要 isinstance 检查 config 是否为 FSDP2Config——
        # 类型标注 `FSDP2Config` 仅声明设计契约，实际调用链可能通过
        # DistributedStrategyConfig 联合类型传入，实现时需要防御性类型检查。
        self.config = config
        self.device_mesh = mesh.device_mesh
        self.device = mesh.device          # 此前缺失，CPU offload 链路依赖此字段

    def parallelize(self, model, tp_shard_plan=None, tp_grad_info: dict | None = None):
        """应用 FSDP2 包裹。

        此时的 model 已经经过了:
        - ShardingPlanner.plan() → TP/CP/SP DTensor 分片
        - apply_sharding_plan() → _local_params_context 解包 DTensor → plain
        - build_tp_grad_info(plan, tp_mesh) → tp_grad_info（TP 梯度旁路）
        - fully_shard(meta) → to_empty → load_base_model（canonical meta 链路）

        FSDP2 在 DP 维度上再做一层包裹:
        - 管理 DP 维度的 all-gather / reduce-scatter
        - 管理参数分片的 CPU offload / 混合精度
        - 感知参数 TP placement（从 tp_grad_info 旁路，写入
          _orig_dtensor_placements / _orig_dtensor_mesh，保持
          _orig_param_is_dtensor=False；详见 §4.4 与 05 §6.7.2）
        """
        # 早期退出：检查 DP 子 mesh 而非全局 world_size
        # 当没有 DP 维度时（如纯 TP 单机），无需 FSDP2 包裹
        # P1 修复：补轴存在守卫——无 dp_shard_cp 轴时（如纯 DDP 单轴 mesh）
        # 返回 model 不包裹，避免 KeyError。
        names = self.device_mesh.mesh_dim_names
        if "dp_shard_cp" not in names:
            # 无 DP shard 轴：可能是纯 TP/CP 拓扑，FSDP2 无事可做
            return model
        dp_mesh = self.device_mesh["dp_shard_cp"]
        if dp_mesh.size() == 1:
            return model

        fsdp2_strategy_parallelize(
            model,
            dp_mesh=dp_mesh,
            tp_shard_plan=tp_shard_plan,
            tp_grad_info=tp_grad_info,
            mp_policy=self.config.mp_policy,
            offload_policy=self.config.offload_policy,
            sequence_parallel=self.config.sequence_parallel,
            activation_checkpointing=self.config.activation_checkpointing,
            enable_async_tensor_parallel=self.config.enable_async_tensor_parallel,
            enable_compile=self.config.enable_compile,
            enable_fsdp2_prefetch=self.config.enable_fsdp2_prefetch,
            backward_prefetch_depth=self.config.backward_prefetch_depth,
            forward_prefetch_depth=self.config.forward_prefetch_depth,
            reshard_after_forward=self.config.reshard_after_forward,
        )
        return model
```

### 4.2 `fsdp2_strategy_parallelize()` 完整签名

```python
# components/distributed/parallelizer.py

def fsdp2_strategy_parallelize(
    model: nn.Module,
    *,
    dp_mesh: DeviceMesh,
    tp_shard_plan: ShardingPlan | None = None,
    tp_grad_info: dict | None = None,
    mp_policy: MixedPrecisionPolicy | None = None,
    offload_policy: CPUOffloadPolicy | None = None,
    reshard_after_forward: bool = True,
    activation_checkpointing: bool | str = False,
    sequence_parallel: bool = False,
    enable_async_tensor_parallel: bool = False,
    enable_compile: bool = False,
    enable_fsdp2_prefetch: bool = False,
    backward_prefetch_depth: int = 1,
    forward_prefetch_depth: int = 1,
) -> nn.Module:
    """FSDP2 + TP 联合分片的主入口。

    在 DTensor TP/CP 分片已应用（apply_sharding_plan）、_local_params_context
    已解包、build_tp_grad_info 已构造 tp_grad_info 之后调用，
    在 DP 维度上应用 fully_shard + activation checkpointing。

    Args:
        model: 已经过 DTensor 分片并解包为 plain local tensor 的模型
        dp_mesh: DP 维度的 DeviceMesh（如 device_mesh["dp_shard_cp"]）
        tp_shard_plan: TP 分片计划（可选，保留用于 AC/SP 决策）。默认 None——
            与 01 §8.3 调用 `parallelize(model, tp_grad_info=...)` 不传
            tp_shard_plan 一致；TP 分片已由 apply_sharding_plan 应用到参数。
        tp_grad_info: TP 梯度旁路 {fqn: (tp_placement, tp_mesh)}，
            由 build_tp_grad_info(plan, tp_mesh) 构造（05 §6.7.1）。
            FSDP2 据此感知参数的 TP placement：写入 _orig_dtensor_placements /
            _orig_dtensor_mesh，保持 _orig_param_is_dtensor=False（生产模式
            参数已是 plain，无 DTensor 可读）；_get_base_spmd_placements 据此
            返回 DeviceMesh.concatenate([dp_mesh, tp_mesh]) +
            [Replicate(DP), tp_placement]（05 §6.7.2）。
        mp_policy: FSDP2 混合精度策略
        offload_policy: CPU offload 策略
        reshard_after_forward: 是否在 forward 后 reshard 参数
        activation_checkpointing: 激活检查点配置（bool/str），与
            FSDP2Config.activation_checkpointing 字段类型一致。取值：
            False=不开、True/"full"=全开、"selective"=选择性。

    Returns:
        包裹后的模型（FSDP2 + AC 已应用）
    """
    # ── 前置校验 ──
    # apply_sharding_plan 已完成：参数已 DTensor 分片 + _local_params_context 解包为 plain
    # local tensor（见 05 §4.4.1）。此处只负责 DP 维度的 fully_shard 包裹 + AC。
    for name, param in model.named_parameters():
        if isinstance(param, DTensor):
            raise RuntimeError(
                f"fsdp2_strategy_parallelize: 参数 {name} 仍为 DTensor——"
                "必须先经 apply_sharding_plan + _local_params_context 解包。"
            )

    tp_enabled = tp_grad_info is not None and bool(tp_grad_info)

    fully_shard_kwargs = dict(
        mesh=dp_mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        reshard_after_forward=reshard_after_forward,
    )
    if tp_enabled:
        # 非 stock API：tp_grad_info 由 FSDP2 写入 _orig_dtensor_placements /
        # _orig_dtensor_mesh，使 _get_base_spmd_placements 推导 DP+TP 联合 grad group
        # （05 §6.7.2）。需要 hyper-parallel 的 FSDP2 fork / 上游 PR 支持。
        fully_shard_kwargs["tp_grad_info"] = tp_grad_info

    # ── Step 1: 定位 transformer layer 列表 ──
    # 启发式：优先 model.model.layers（HF Llama/Qwen 系），回退 model.transformer.h
    # （GPT 系），再回退 model.layers。未命中则只包裹 root unit（小模型 / 非标准结构）。
    layers = None
    for attr_path in ("model.layers", "transformer.h", "layers"):
        obj = model
        ok = True
        for seg in attr_path.split("."):
            obj = getattr(obj, seg, None)
            if obj is None:
                ok = False
                break
        if ok and isinstance(obj, (list, tuple)) and len(obj) > 0:
            layers = obj
            break

    # ── Step 2: per-layer AC + fully_shard ──
    if layers is not None:
        for block in layers:
            if activation_checkpointing:
                _apply_activation_checkpoint(
                    block,
                    mode=activation_checkpointing,
                    sequence_parallel=sequence_parallel,
                    tp_grad_info=tp_grad_info if tp_enabled else None,
                )
            fully_shard(block, **fully_shard_kwargs)

    # ── Step 3: root unit（embed + lm_head 共享一个 FSDP unit，tied weights 场景） ──
    # root unit 必须最后包裹（FSDP2 要求内层先包裹）。tied weights 归一化见 05 §6.7.1。
    fully_shard(model, **fully_shard_kwargs)

    # ── Step 4: 可选 prefetch（复用自研 fully_shard 现有 API，非 fork 新增） ──
    if enable_fsdp2_prefetch and backward_prefetch_depth > 0:
        _wire_fsdp2_prefetch(
            model, backward_prefetch_depth, forward_prefetch_depth
        )

    # enable_async_tensor_parallel / enable_compile 暂为占位——前者需要 TP all-gather
    # 与 compute 的 stream overlap（依赖 collectives 层 async_op），后者见 §torch.compile
    # 专项（本期搁置）。两者当前仅记录意图，不改变包裹结果。
    return model


def _apply_activation_checkpoint(
    module: nn.Module, *, mode, sequence_parallel: bool, tp_grad_info
) -> None:
    """对 transformer block 应用激活检查点。

    mode 取值与 FSDP2Config.activation_checkpointing 一致：
      False / None → 不开；
      True / "full" → 整 block checkpoint；
      "selective" → 仅对 attention/MLP 子模块 checkpoint（ heuristic：类名含
        Attention/Era/MLP，或可通过 05 ARCH_OVERRIDES 指定）。
    """
    if mode in (False, None):
        return
    if mode in (True, "full"):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_module, apply_activation_checkpointing_checkpoint_wrapper,
        )
        # checkpoint_module 对整个 block 重写 forward 为 AC 版本
        checkpoint_module(module)
        return
    if mode == "selective":
        # 选择性 AC：遍历子模块，对命中规则的子模块单独 checkpoint
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_module,
        )
        _SELECTIVE_SUBSTR = ("Attention", "MLP", "DecoderLayer", "EncoderLayer")
        for name, sub in module.named_children():
            cls_name = type(sub).__name__
            if any(s in cls_name for s in _SELECTIVE_SUBSTR):
                checkpoint_module(sub)
        return
    raise ValueError(f"_apply_activation_checkpoint: unknown mode={mode!r}")


def _wire_fsdp2_prefetch(
    model: nn.Module, backward_depth: int, forward_depth: int
) -> None:
    """为 FSDP2 block 注册 prefetch（复用自研 fully_shard 现有 API，无需 fork）。

    `core/fully_shard/api.py` 的 `HSDPModule` 已提供
    `set_modules_to_forward_prefetch(modules)` /
    `set_modules_to_backward_prefetch(modules)`——传入按执行顺序排序的
    FSDP block 列表即可实现多级 prefetch（hsdp_scheduler 按列表驱动
    all-gather 预取）。depth 字段映射为"取有序 block 列表的前 depth 层"，
    depth<=0 的方向跳过。
    """
    if backward_depth <= 0 and forward_depth <= 0:
        return
    # 收集 FSDP 包裹的 transformer block（保持 forward 执行顺序）
    blocks = [
        m for m in model.modules()
        if type(m).__name__ in ("FSDPModule", "HSDPModule") and m is not model
    ]
    if forward_depth > 0:
        model.set_modules_to_forward_prefetch(blocks[:forward_depth])
    if backward_depth > 0:
        # backward 逆序：取执行序的末尾 depth 层
        model.set_modules_to_backward_prefetch(blocks[-backward_depth:])
```

### 4.3 层叠架构：DTensor (TP/CP) → FSDP2 (DP)

```
┌─────────────────────────────────────────────────┐
│ Layer 3: FSDP2 (DP 维度)                         │
│   - all-gather / reduce-scatter                  │
│   - 混合精度 / CPU offload                        │
│   - 参数分片 (dp_shard 维度)                      │
├─────────────────────────────────────────────────┤
│ Layer 2: DTensor (TP/CP 维度)                    │
│   - PrecompiledBoundary (编译期通信规划)           │
│   - _local_params_context (零拷贝参数替换)          │
│   - Shard(0)/Shard(1)/Replicate() placements      │
├─────────────────────────────────────────────────┤
│ Layer 1: nn.Module (模型结构)                     │
│   - 原始 HF transformer 实现                      │
└─────────────────────────────────────────────────┘
```

**关键**：DTensor 分片先应用（在 `apply_sharding_plan()` 中），FSDP2 包裹后应用（在 `fsdp2_manager.parallelize()` 中）。FSDP2 管理的是 DP 维度，与 DTensor 的 TP/CP/SP 维度正交——两者不冲突，共同构成完整的 4D 并行。

### 4.3 YAML 配置到 Mesh 构建的完整路径

```yaml
distributed:
  tp: 4
  cp: 2
  pp: 1
  strategy: fsdp2
  sequence_parallel: true
  activation_checkpointing: selective

  # FSDP2 高级选项
  mp_policy:
    _target_: torch.distributed.fsdp.MixedPrecisionPolicy
    param_dtype: bfloat16
    reduce_dtype: float32
  reshard_after_forward: true
  defer_fsdp_grad_sync: true
```

```
YAML → RecipeConfig
    → create_distributed_setup_from_config()
        → ParallelismSizes(tp_size=4, cp_size=2, pp_size=1)
        → MeshContext.build(FSDP2Config, sizes)
            → DeviceMesh("cuda", (8,2,4), ("dp_shard_cp","cp","tp"))
        → DistributedSetup(mesh_context, strategy_config, ...)
    → mesh = distributed_setup.mesh_context
    → sharding_planner.plan(model, mesh.device_mesh, ...)
    → FSDP2Manager(strategy_config, mesh)          # 收 MeshContext（C6 canonical）
    → model, tp_grad_info = apply_sharding_plan(model, plan, mesh)   # 内部调 build_tp_grad_info
    → fsdp2_manager.parallelize(model, tp_grad_info=tp_grad_info)
```

---

### 4.4 FSDP2 包裹粒度

FSDP2 采用 **per-transformer-block + root unit** 的包裹策略，而非对整个模型做单一 `fully_shard`：

```python
# 包裹粒度：每个 transformer layer 一个 FSDP unit
for layer_id, block in enumerate(model.model.layers):
    fully_shard(
        block,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
        offload_policy=offload_policy,
        tp_grad_info=tp_grad_info,        # TP 梯度旁路，见 05 §6.7
    )

# root unit：embed + lm_head 共享一个 FSDP unit（tied weights 场景）
fully_shard(
    model,
    mesh=dp_mesh,
    mp_policy=mp_policy,
    reshard_after_forward=reshard_after_forward,
    offload_policy=offload_policy,
    tp_grad_info=tp_grad_info,
)
```

`fully_shard()` 是 FSDP2 的函数式 API（不是旧版 `FSDP(...)` 构造函数）。
它直接将模块的参数在 DP mesh 上做 all-gather / reduce-scatter 分片管理。
`tp_grad_info` 作为关键字参数传入：FSDP2 内部将其写入参数的
`_orig_dtensor_placements` / `_orig_dtensor_mesh`（保持
`_orig_param_is_dtensor=False`，因为生产模式参数已被 `_local_params_context`
解包为 plain），`_get_base_spmd_placements` 新增分支返回
`DeviceMesh.concatenate([dp_mesh, tp_mesh])` + `[Replicate(DP), tp_placement]`，
使 FSDP2 能据 TP placement 推导 DP+TP 联合 grad group（交叉引用 05 §6.7.2）。

**root unit tied weights 语义**：当 embed 与 lm_head 共享权重（tied）时，
root unit 必须是包含两者的最小公共 FSDP unit，且 `tp_grad_info` 中 tied
参数的 TP placement 必须一致——否则同一物理参数会被注册两次导致梯度 all-reduce
重复。tied 对的来源是 `ShardingPlan.tied_pairs`（由 ShardingPlanner 从模型
weight tying 检测填入，05 §4.3）；归一化（placement 不一致时取较细分片，
Shard 优先于 Replicate）由 05 §6.7.1 `build_tp_grad_info(plan, tp_mesh,
tied_pairs=plan.tied_pairs)` 实现——06 仅声明 root unit 契约，不重复归一化
逻辑。`build_tp_grad_info` 由 `apply_sharding_plan` 内部调用（05 §4.1
canonical），`fsdp2_strategy_parallelize` 接收已构造的 `tp_grad_info` 作为
输入参数。P2: tied_pairs 具体来自 `plan.tied_pairs`（非解包后的另一来源），
与 05 §4.3 tied_pairs 检测逻辑对齐。
P2: `fsdp2_strategy_parallelize` 不直接调用 `build_tp_grad_info`（此描述
已在 P1-6 修复中确认正确，此处仅加注说明）。

### 4.5 EP/DP 拓扑 (MoE)

在 MoE 场景下，expert 参数需要额外的 Expert Parallel (EP) 维度：

```python
# EP 从 DP 切出：ep_size 整除 dp_size（见 _validate_parallelism_sizes）
# 总 world_size = dp_size * tp_size * cp_size（dp 内部再切 ep）
# moe_mesh 不再从主 mesh submesh(["ep_shard"]) 取（主 mesh 原无 ep_shard 轴），
# 而是在 MeshContext.build 中构造主 mesh 时携带 ep_shard 子轴并赋给 moe_mesh 字段。
moe_mesh = mesh_context.moe_mesh   # 由 build() 赋值，DeviceMesh("ep_shard",)

# 每个 expert 在一个 EP group 内复制，跨 EP group 分片
for expert in moe_layer.experts:
    fully_shard(expert, mesh=moe_mesh, ...)
```

`_validate_parallelism_sizes` 负责校验 ep/dp 配置的合法性（调用点与定义 arity 一致：6 参 `(tp, cp, pp, ep, dp, world_size)`，由 `MeshContext.build` 传入从 `ParallelismSizes` 读取的 `ep`/自动推导的 `dp`）：

```python
def _validate_parallelism_sizes(tp, cp, pp, ep, dp, world_size):
    # 校验所有并行维度的乘积等于 world_size。
    # 注意：此处的 dp **不含** ep（与 build() 一致）——EP 从 dp 切出，
    # per_ep = dp // ep，故 world_size = tp*cp*pp*ep*dp（dp 排除 ep）成立。
    # 这与"EP 从 DP 切出"并不矛盾：dp 是切出 EP 之前的 DP 总量。
    #
    # **与 _build_fsdp2_mesh 的 dp 语义关系**（P1 修复统一口径）：
    # - _validate 的 dp **排除 ep**（dp = per_ep * ep 不成立，dp 是 EP 切出前的 DP 总量）；
    # - _build_fsdp2_mesh 的 dp_total / dp_shard **含 ep**（dp_shard = dp_total / dp_replicate，
    #   dp_shard = per_ep * ep_size，ep_shard 是 dp_shard 的子轴）。
    # 两者口径不同但不矛盾：_validate 校验的是"用户声明的 dp（排除 ep）"，
    # _build_fsdp2_mesh 构造的是"mesh 轴的 dp_shard（含 ep）"。校验通过后，
    # dp_shard % ep_size == 0 由 _build_fsdp2_mesh 的 assert 保证（因 dp % ep == 0
    # 且 dp_shard = dp * ep / dp_replicate，故 dp_shard % ep == 0 恒成立）。
    # 不存在"校验通过、构造 AssertionError"盲区。
    product = tp * cp * pp * ep * dp
    if product != world_size:
        raise ValueError(
            f"tp({tp}) * cp({cp}) * pp({pp}) * ep({ep}) * dp({dp}) = {product} "
            f"!= world_size({world_size})"
        )
    # EP 从 DP 切出：ep_size 必须 <= dp_size 且 dp_size 能被 ep_size 整除
    if ep > 1:
        if ep > dp:
            raise ValueError(f"ep_size({ep}) > dp_size({dp})，EP 必须从 DP 切出")
        if dp % ep != 0:
            raise ValueError(f"dp_size({dp}) % ep_size({ep}) != 0，EP 必须整除 DP")
    # 额外校验：num_attention_heads % tp == 0 等模型维度约束
    # P2: 模型维度校验（heads % tp, num_kv_heads % cp 等）由
    # ShardingPlanner.plan() 单独执行（05 §4.2 validate_model_compatibility），
    # _validate_parallelism_sizes 仅校验拓扑一致性（world_size 乘积），不重复模型维度校验。
```

**EP 拓扑 canonical 约定**：EP 不与 DP 正交，而是**从 DP 切出**——`ep_size` 必须整除 `dp_size`，EP group 完全落在 DP group 内部。这样 MoE 的 expert all-to-all 不会跨 DP 边界，且 `MeshContext.build` 在构建主 mesh 时即在同一 DeviceMesh 上切出 `ep_shard` 子轴，`moe_mesh` 在 `build()` 末尾由 `ctx.device_mesh["ep_shard"]` 赋值（此前该字段从不赋值）。

### 4.6 HYBRID_SHARD 双轴 Mesh

FSDP2 支持 HYBRID_SHARD 模式，在节点内 shard、节点间 replicate：

```python
# HYBRID_SHARD: dp_replicate（节点间） + dp_shard（节点内）
# 例如 4 节点 × 8 GPU = 32 GPU, dp_replicate=4, dp_shard=8
dp_replicate_size = 4  # 跨节点 replicate
dp_shard_size = 8       # 节点内 shard

# 构建双轴 DP mesh
dp_mesh_2d = torch.arange(world_size).reshape(dp_replicate_size, dp_shard_size)
dp_device_mesh = DeviceMesh("cuda", dp_mesh_2d, ("dp_replicate", "dp_shard"))

# fully_shard 在 dp_shard 维度做 all-gather/reduce-scatter，
# 不跨 dp_replicate 维度通信 → 减少跨节点通信量
# P2: 此示例为独立演示 HYBRID_SHARD 概念，轴名用 dp_shard；
# canonical 主 mesh 统一使用 dp_shard_cp（详见下方命名衔接说明）。
fully_shard(model, mesh=dp_device_mesh["dp_shard"], ...)
```

> **与 `_build_fsdp2_mesh` 的命名衔接**：上例为手动构建的示意 mesh，轴名为
> `("dp_replicate", "dp_shard")`。`MeshContext._build_fsdp2_mesh` 在
> HYBRID_SHARD 分支（`dp_replicate_size>1`）产出的主 mesh 轴序为
> `("dp_replicate", "dp_shard_cp", "cp", "tp")`——节点内 shard 轴名为
> `dp_shard_cp`（与普通 FSDP2 保持一致，便于 `FSDP2Manager.parallelize`
> 统一通过 `self.device_mesh["dp_shard_cp"]` 取得 FSDP 子 mesh）。
> `dp_size` property 据此返回 `dp_replicate.size() * dp_shard_cp.size()`。

---

## 5. _local_params_context：生产模式零拷贝

### 5.1 设计动机

在 apply_sharding_plan() 的生产模式下，模型参数已经是 DTensor。但在 forward 内部，我们只需要参数的**本地分片**（`DTensor._local_tensor`），不需要 DTensor 的 placement 元数据和 dispatch 开销。

采用 **build 期一次性解包**策略——在 `fully_shard` 之前将 DTensor 参数永久替换为 `_local_tensor`，而非每次 forward 使用 context manager 切换：

```python
# components/distributed/dtensor_utils.py
#
# Canonical 实现位于 components/distributed/sharding/apply.py（05 §4.4.1），
# 本模块仅作 re-export 入口，避免重复定义导致两份实现漂移
# （历史问题：06 曾在此处独立实现一份，漏掉 requires_grad 继承，
#  导致参数冻结静默失效——已收敛为单一权威定义）。

from .sharding.apply import _local_params_context, _set_param_by_path  # noqa: F401

__all__ = ["_local_params_context", "_set_param_by_path"]
```

> **完整实现见 [05 §4.4.1](05_dual_mode_dtensor_parallel_strategy.md)**。
> 实现要点（摘录，权威定义以 05 为准）：
> - 遍历 `model.named_parameters()`，对 `isinstance(param, DTensor)` 的参数，
>   以 `nn.Parameter(param._local_tensor, requires_grad=param.requires_grad)`
>   永久替换——`_local_tensor` 与 DTensor 共享存储（零拷贝），`requires_grad`
>   继承自原 DTensor 参数（冻结语义不丢）。
> - 路径式赋值用 `_set_param_by_path` 走 `register_parameter`，沿点分 FQN
>   定位到真正的父模块（`object.__setattr__(model, "layers.0...", ...)` 只会
>   在 model 上设一个怪属性，不会替换子模块参数）。
> - 返回 `{fqn: placements}` 仅作解包前后的等价性 sanity check；
>   `tp_grad_info` 的 canonical 数据来源是 `ShardingPlan`
>   （`build_tp_grad_info(plan, tp_mesh)`，05 §6.7.1），而非此返回值。

### 5.2 在 apply_model_infrastructure（_build_model 内部调用）中的使用时机

```python
# apply_model_infrastructure（_build_model 内部调用）中的核心编排调用顺序
# canonical meta 链路：fully_shard 在 to_empty 之前（meta 上包裹 FSDP，
# 物化 sharded 参数，再 load 写入），与 P0-2 对齐。

def _build_model():
    # ...
    # Step 7: apply_sharding_plan → DTensor 分片 + 生产模式解包
    #   apply_sharding_plan 内部 Phase C 调用 _local_params_context(model)，
    #   并内部调 build_tp_grad_info，返回 (model, tp_grad_info)（与 05 §4.1 canonical 一致）
    model, tp_grad_info = apply_sharding_plan(model, plan, mesh.device_mesh)

    # Step 8: fully_shard(meta) → 在 meta 模型上包裹 FSDP2
    #   必须在 to_empty / load 之前——FSDP2 在 meta 上注册 sharded 参数，
    #   物化后由 load_base_model 写入实际权重
    fsdp2_manager.parallelize(model, tp_shard_plan=plan, tp_grad_info=tp_grad_info)

    # Step 9: to_empty → 物化 sharded 参数（FSDP2 canonical 顺序）
    model.to_empty(device=mesh.device)

    # Step 10: load_base_model → 权重写入已 sharded 的参数
    load_base_model(model, mesh.device, pretrained_path, adapter=..., mesh=mesh.device_mesh)
    # ...
    return model
```

### 5.3 零拷贝验证

```python
# _local_tensor 与 DTensor 共享底层存储
dt = distribute_tensor(torch.randn(16, 64), mesh, [Shard(0)])
local = dt._local_tensor

# 零拷贝：修改 local 会影响 dt（反之亦然）
local[0, 0] = 99.0
assert dt[0, 0] == 99.0  # passes
# 不触发任何 NCCL all-gather ！
```

---

## 6. Grad-Accumulation Helper 签名

以下 6 个 helper 由 FSDP2Manager 的 `defer_fsdp_grad_sync` 与训练循环的
梯度累积路径共同调用，归属 `components/training/grad_accum.py`（与 03 §7.1
签名一致）。此前 06 全文无签名无引用，此处补齐以支撑 03 训练循环的调用点。

```python
# components/training/grad_accum.py
#
# P2: FSDPModule 通过 torch.distributed.fsdp2.FSDPModule 导入，
# 此处各 helper 中的 `isinstance(mp, FSDPModule)` 依赖该 import。
# 实际实现时在文件顶部添加：
#   from torch.distributed.fsdp2 import FSDPModule

def get_sync_ctx(
    model_parts: list[nn.Module],
    *,
    is_optim_step: bool,
    defer_fsdp_grad_sync: bool = False,
) -> object:
    """返回 forward 期间的上下文管理器（用于 defer_fsdp_grad_sync 场景）。

    - is_optim_step 且非 defer：返回 FSDP2 的 `sum_full_grads` 上下文
      （最后一个 microbatch 时聚合 full grad）。
    - defer_fsdp_grad_sync=True：返回 no-op 上下文，并标记各 FSDPModule
      进入 deferred sync 模式（grad 不立即 all-reduce）。
    - 非 optim step：返回 nullcontext()。

    签名与 03 §7.1 canonical 一致。"""
    # Canonical implementation see 03 §7.1. 06 declares signature only for module-level completeness.

def prepare_for_grad_accumulation(model_parts: list[nn.Module]) -> None:
    """梯度累积开始前的准备：
    1. opt.zero_grad 由调用方在 Phase 3 统一执行（本函数不重复）；
    2. 对每个 FSDPModule 调 `set_requires_gradient_sync(False)`，
       进入 deferred-sync 模式；
    3. 记录 `_grad_accum_state`（用于 final backward 时还原）。

    签名与 03 §7.1 canonical 一致（无 num_microbatches/sync_grads 必填参）。"""
    # Canonical implementation see 03 §7.1. 06 declares signature only for module-level completeness.

def prepare_for_final_backward(model_parts: list[nn.Module]) -> None:
    """最后一个 microbatch 反向前的准备：
    1. 遍历所有 FSDPModule 调 `set_requires_gradient_sync(True)`，
       允许 backward 末尾触发 DP all-reduce；
    2. PP 多 stage 时还要在各 stage 间挂上 send/recv 钩子
       （PP 的 backward 由最后一 stage 触发）。

    签名与 03 §7.1 canonical 一致（无 num_label_tokens 必填参；
    loss scale 由 scale_grads_and_clip_grad_norm 单独处理）。"""
    # Canonical implementation see 03 §7.1. 06 declares signature only for module-level completeness.

def prepare_after_first_microbatch(
    model_parts: list[nn.Module],
) -> None:
    """第一个 microbatch 前向后的调整：启用 FSDP2 forward prefetch、
    触发参数 reshard（若 reshard_after_forward=True）。"""
    # Canonical implementation see 03 §7.1. 06 declares signature only for module-level completeness.

def set_requires_gradient_sync(
    model_parts: list[nn.Module],
    is_last: bool,
) -> None:
    """逐 part 设置 FSDP2 梯度同步开关（中间 microbatch 关，最后一个开）。
    等价于 `for mp in model_parts: if FSDPModule: mp.set_requires_gradient_sync(is_last)`。

    形参名 `is_last`（非 requires_sync），与 03 §7.1 canonical 一致。

    注意：canonical 实现位于 03 §7.1；06 仅声明此签名，用于模块的公开 API 表面文档。"""
    # Canonical implementation see 03 §7.1. 06 declares signature only for module-level completeness.

def scale_grads_and_clip_grad_norm(
    model_parts: list[nn.Module],
    max_norm: float,
    num_label_tokens: int | None = None,
) -> float:
    """对累积梯度按 1/num_label_tokens 缩放并计算/裁剪 grad norm，返回 grad_norm。

    - num_label_tokens 非 None（非 PP 场景）：对每个参数 grad 除以
      num_label_tokens，将 CE sum 还原为 token-mean。
    - num_label_tokens 为 None（PP 场景）：跳过缩放（PP 各 stage 的
      token 归一化由 calculate_loss/calculate_loss_pp 统一处理）。

    参数顺序 (model_parts, max_norm, num_label_tokens) 与 03 §7.1 调用点
    一致；第三参 canonical = `num_label_tokens: int | None = None`
    （03 PP 传 None）。"""
    # Canonical implementation see 03 §7.1. 06 declares signature only for module-level completeness.
```

---

## 7. 配置示例

```yaml
recipe: FinetuneRecipe

# ── 分布式拓扑 ──
distributed:
  tp: 4                                      # Tensor Parallel
  cp: 2                                      # Context Parallel
  pp: 1                                      # Pipeline Parallel
  strategy: fsdp2                            # "fsdp2" | "ddp" | "megatron_fsdp"
  sequence_parallel: true
  activation_checkpointing: selective         # false | true | "full" | "selective"

  # FSDP2 混合精度
  # 注：_target_ 指向 FSDP1 命名空间路径 torch.distributed.fsdp.MixedPrecisionPolicy，
  # FSDP2 同名类位于 torch.distributed.fsdp2 ——实现时按 strategy_config 类型
  # 解析为对应类（FSDP2Config 时强制使用 fsdp2.MixedPrecisionPolicy），
  # 此处保留 FSDP1 路径仅为 YAML 兼容历史配置，新配置应直接写
  # torch.distributed.fsdp2.MixedPrecisionPolicy。
  mp_policy:
    _target_: torch.distributed.fsdp.MixedPrecisionPolicy
    param_dtype: bfloat16
    reduce_dtype: float32

  # FSDP2 高级
  reshard_after_forward: true
  defer_fsdp_grad_sync: true

model:
  _target_: hyper_parallel.HyperAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: Qwen/Qwen3.5-4B
```

---

## 8. 与现有文档的关系

| 文档 | 覆盖内容 | 本文档的关系 |
|------|---------|------------|
| 01 §8 | `instantiate_infrastructure()` — 创建 FSDP2Manager + ShardingPlanner | 本文档 §4 展开 FSDP2Manager 的实现细节 |
| 01 §6.3 | `_build_model()` — ④.4.5.10 `fsdp2_manager.parallelize()` | 本文档 §4.2 展开 `parallelize()` 的实现 |
| 05 §10 | FSDP2 与 DTensor 的关系概述 | 本文档 §4.2 给出层叠架构图 |
| 05 §6.3 | 生产模式 forward 包装 | 本文档 §5 给出 `_local_params_context` 的完整实现 |
