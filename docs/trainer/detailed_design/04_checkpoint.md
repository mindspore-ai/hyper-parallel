# Hyper-Parallel Checkpoint 模块详细设计

> 参考实现：[AutoModel `components/checkpoint/checkpointing.py`](../../../auto_model/Automodel/nemo_automodel/components/checkpoint/checkpointing.py)
> 上下文设计：[dual_mode_dtensor_parallel_strategy.md](../dual_mode_dtensor_parallel_strategy.md)

---

## 1. 模块职责

提供完整的 checkpoint 保存/恢复体系——DCP 切分保存（含 DTensor 元数据）、HF safetensors 导出、异步保存、SIGTERM 故障恢复、Stateful 组件追踪。

### 核心文件

| 文件 | 职责 |
|------|------|
| `hyper_models/components/checkpoint/checkpointing.py` | `Checkpointer` 核心类 (new file, to be created) |
| `hyper_models/components/checkpoint/config.py` | `CheckpointingConfig` (new file, to be created) |
| `hyper_models/components/checkpoint/stateful_wrappers.py` | `ModelState` / `OptimizerState` wrapper (new file, to be created) |
| `hyper_models/components/checkpoint/conversion_mapping.py` | HF key ↔ 模型 FQN 映射 + `WeightConverter` (new file, to be created) |
| `hyper_models/components/checkpoint/_backports/hf_storage.py` | `_HuggingFaceStorageWriter` / `_HuggingFaceStorageReader` (new file, to be created) |
| `hyper_models/components/checkpoint/addons.py` | `ConsolidatedHFAddon` / `PeftAddon` (new file, to be created) |

### 涉及删除的旧代码

| 旧代码 | 替代方案 |
|--------|---------|
| `hyper_parallel/core/distributed_checkpoint/` (大部分) | 保留 `planner.py` / `standard_planner.py`（增加 DTensor metadata），删除 `api.py`（合并到 Checkpointer） |
| `hyper_parallel/trainer/callbacks/base.py` — `CheckpointCallback` | 移到 `hyper_models/components/checkpoint/checkpointing.py` |
| `hyper_parallel/trainer/callbacks/base.py` — `SafetensorsExportCallback` | 移到 `Checkpointer.save_model()` 内 |
| `hyper_parallel/core/distributed_checkpoint/offline_transform.py` | 移到 `hyper_models/components/checkpoint/_backports/consolidate_hf_safetensors.py` |

---

## 2. 总入口调用时序：从 `recipe.setup()` 到 Checkpoint 保存/恢复

Checkpoint 模块在两条调用路径上工作——**初始化**（`setup()` 中创建 Checkpointer）和**运行时**（训练循环中保存/加载）。

```
main() → recipe.setup(cfg)                                           # 01_hf_compatibility_layer.md §4
│
├─④.7 self.checkpointer = cfg.checkpoint.build(                      # §3/§4: 初始化路径
│       dp_rank=..., tp_rank=..., pp_rank=..., moe_mesh=...)
│   │
│   ├─ cfg.checkpoint                                                # RecipeConfig cached_property (§8)
│   │   → CheckpointingConfig(                                       # §3: 类型化配置
│   │         checkpoint_dir="outputs/",
│   │         model_save_format="safetensors",
│   │         save_consolidated="final",
│   │         is_async=True,
│   │         model_repo_id="Qwen/Qwen3.5-0.8B",    # 从 model 段派生
│   │         is_peft=False,                         # 从 peft 段派生
│   │         ...)
│   │
│   └─ CheckpointingConfig.build(dp_rank, tp_rank, pp_rank, moe_mesh)
│       → Checkpointer(config, dp_rank, tp_rank, pp_rank, moe_mesh)  # §4: Checkpointer 实例
│           ├─ 初始化 StorageWriter / StorageReader (safetensors / torch_save)
│           ├─ 注册 Addons (ConsolidatedHFAddon / PeftAddon)
│           └─ 异步 stager (if is_async)
│
├─④.13 self.load_checkpoint(restore_from)                            # 断点续训恢复（§8 Recipe 方法）
│   │
│   └─ checkpointer.load_model(self.model_parts, model_path)         # §5.3: DCP resume（仅 resume 语义）
│       ├─ ModelState(model_parts) → state_dict()                    #   遍历所有 PP part
│       ├─ adapter.to_hf → dcp.load → adapter.from_hf                #   key 对齐 checkpoint
│       └─ model_state.load_state_dict(state_dict)
│   │
│   └─ checkpointer.load_optimizer(self.model_parts, optimizers, ...) # 恢复优化器状态（list，与 03 §3.1 canonical 对齐）
│   └─ load LR scheduler / RNG / DataLoader state                    # 恢复其他组件
│
└─⑤ recipe.run_train_validation_loop()                               # 训练循环
    │
    └─ if is_ckpt_step: self.save_checkpoint(...)                     # 03 §3 BaseRecipe: 遍历 __state_tracked
        │
        ├─ checkpointer.save_model(model, f"{path}/model")            # §4.2: 保存路径
        │   │
        │   ├─ ModelState(model) → state_dict()                       # §5: 获取 DTensor state dict
        │   │
        │   ├─ StateDictAdapter.to_hf(state_dict)                     # §8: 内部 key → HF key
        │   │
        │   ├─ _materialize_non_contiguous(state_dict)                # safetensors 要求连续视图
        │   │
        │   ├─ get_fqn_to_file_index_mapping(state_dict)              # 构建分片索引
        │   │
        │   ├─ Addon.pre_save()                                       # config.json / tokenizer.json
        │   │
        │   ├─ dcp.save(state_dict, checkpoint_id=path,               # §4.2: DCP 写入
        │   │          storage_writer=_HuggingFaceStorageWriter(...))
        │   │   │
        │   │   └─ _extract_dtensor_metadata(state_dict) → sidecar JSON
        │   │       # §6: DTensor 元数据写入 .dtensor_metadata.json（可观测性，不进 DCP SavePlan）
        │   │
        │   ├─ Addon.post_save()
        │   │
        │   └─ consolidate_safetensors_files_on_every_rank(...)       # §4.2: 合并导出 HF safetensors
        │       └─ 每 rank 并行写自己的 shard → 最终合并为 .safetensors 文件
        │
        └─ checkpointer.save_optimizer(model, optimizers, f"{path}/optimizer")    # §5: OptimizerState → DCP（与 _state_path 同源；optimizers 为 list，03 §3.1 canonical）

── 模型初始化路径（from_pretrained 内部）──

HyperAutoModelForCausalLM.from_pretrained()                           # 01 §6
└─ _build_model()
    ├─ safetensors_path = _get_hf_safetensors_reference_path(root_dir, model_name)  # 解析 safetensors 路径
    └─ load_base_model(model, device, safetensors_path,              # §5.3 路径1/2（path 已解析）
                       adapter=_get_state_dict_adapter(model), mesh=mesh)
        │
        ├─ 路径1: MoE tensor merging (requires_tensor_merging)
        │   └─ _convert_checkpoint_with_transformers(path, model.config)
        │       → HF WeightConverter 合并分散的 expert 权重
        │       → _load_full_state_dict_into_model(model, state_dict)  # §5.4: 零 NCCL
        │
        └─ 路径2: Safetensors fast path
            ├─ _load_hf_checkpoint_preserving_dtype(path)            # §5.4: 每 rank 并行读
            │   ├─ 读 model.safetensors.index.json → shard 文件列表
            │   └─ safetensors.load_file(shard, device="cpu")         # 全量 CPU state dict
            │
            ├─ StateDictAdapter.from_hf(state_dict, device_mesh=mesh) # HF key → 模型内部 key
            │
            └─ _load_full_state_dict_into_model(model, state_dict)    # §5.4: 零 NCCL 加载
                └─ set_model_state_dict(model, state_dict,             # PyTorch DCP 原生
                       options=StateDictOptions(full_state_dict=True))
                    # 不设 broadcast_from_rank0=True
                    # 每个 rank 独立从全量 state dict 中切出自己的 DTensor shard
                    # 零 NCCL communication
```

**与 01、03 文档的时序衔接**：

```
main()                                  # 01 §4
├─④ recipe.setup(cfg)
│   ├─④.7  checkpointer = ...          # 本文档 §3/§4 (初始化)
│   ├─④.8  model = from_pretrained()   # 01 §6
│   │   └─ load_base_model()           # 本文档 §4.3/§4.4 (权重加载)
│   └─④.13 load_checkpoint()           # 本文档 §8 (断点续训恢复)
└─⑤ run_train_validation_loop()        # 03_training_loop.md §6
    └─ save_checkpoint()               # 本文档 §4.2 (保存)
```

---

## 3. Checkpoint 目录结构

```
{checkpoint_dir}/
├── epoch_0_step_100/
│   ├── model/
│   │   ├── dp_rank_0/
│   │   │   └── __0_0.distcp           ← DCP 切分权重（per-rank shard）
│   │   ├── dp_rank_1/
│   │   │   └── __1_0.distcp
│   │   ├── consolidated/              ← 可选：HF 兼容合并导出
│   │   │   ├── model-00001-of-00002.safetensors
│   │   │   ├── model-00002-of-00002.safetensors
│   │   │   └── model.safetensors.index.json
│   │   ├── .hf_metadata/
│   │   │   ├── config.json            ← 模型配置
│   │   │   └── tokenizer.json         ← Tokenizer 配置
│   │   └── consolidate.sh             ← 离线合并脚本（由 _write_consolidate_script 写入 model/ 目录下）
│   ├── optimizer/
│   │   ├── dp_rank_0/
│   │   │   └── __0_0.distcp           ← DCP optimizer state (per-rank)
│   │   └── dp_rank_1/
│   │       └── __1_0.distcp
│   ├── dataloader/
│   │   └── dataloader_dp_rank_0.pt
│   ├── rng/
│   │   └── rng_dp_rank_0.pt
│   ├── scheduler.pt                   ← LR scheduler state (rank 0)
│   └── extra_state.json               ← global_step + epoch
└── LATEST → epoch_0_step_100          ← 软链接指向最新 checkpoint（存相对路径，见 03 §7.1
                                         ``_update_latest_symlink``；位于 checkpoint_dir 根，
                                         不在 step 目录内部）
```

---

## 4. CheckpointingConfig

> **调用位置**: 时序树 ④.7 — `RecipeConfig.checkpoint` → `CheckpointingConfig(**kwargs)`

```python
# hyper_models/components/checkpoint/config.py

import torch
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class SaveConsolidatedMode(str, Enum):
    FALSE = "false"
    FINAL = "final"
    EVERY = "every"


class SerializationFormat(str, Enum):
    SAFETENSORS = "safetensors"
    TORCH_SAVE = "torch_save"  # DCP .distcp


@dataclass
class CheckpointingConfig:
    """Checkpoint 配置。"""

    # ── 基本设置 ──
    enabled: bool = True
    checkpoint_dir: str | Path = "checkpoints/"

    # ── 保存格式 ──
    model_save_format: str = "safetensors"  # "safetensors" | "torch_save"
    save_consolidated: SaveConsolidatedMode | str = SaveConsolidatedMode.FINAL  # "false" | "final" | "every"
    is_peft: bool = False

    # ── 异步 ──
    is_async: bool = False  # torch >= 2.3.0 才支持

    # ── 模型来源 ──
    model_cache_dir: str | None = None  # reserved for future use
    model_repo_id: str | None = None
    original_model_root_dir: str | None = None  # 基座模型根目录（用于 HF metadata）；set by RecipeConfig bridge in 01

    # ── DCP 恢复 ──
    model_state_dict_keys: list[str] | None = None  # 预并行化 key 快照（DCP 恢复时校验）

    # ── 兼容性 ──
    v4_compatible: bool = False      # 写入旧版格式 config.json
    diffusers_compatible: bool = False  # diffusers 索引文件名格式

    # ── 高级 ──
    dequantize_base_checkpoint: bool | None = None
    skip_task_head_prefixes: list[str] | None = None       # 保存时跳过的前缀
    skip_task_head_prefixes_for_base_model: list[str] | None = None  # base model 加载时跳过的前缀
    single_rank_consolidation: bool = False
    staging_dir: str | None = None
    best_metric_key: str = "default"

    def __post_init__(self):
        # 格式规范化
        if self.model_save_format == "torch_save":
            self._serialization_format = SerializationFormat.TORCH_SAVE
        else:
            self._serialization_format = SerializationFormat.SAFETENSORS

        # save_consolidated 归一化为 Enum（接受 str | bool 输入）
        if isinstance(self.save_consolidated, bool):
            self.save_consolidated = (
                SaveConsolidatedMode.EVERY if self.save_consolidated
                else SaveConsolidatedMode.FALSE
            )
        elif isinstance(self.save_consolidated, str):
            self.save_consolidated = SaveConsolidatedMode(self.save_consolidated)

        # PEFT 强制 safetensors 格式
        if self.is_peft:
            self._serialization_format = SerializationFormat.SAFETENSORS
            self.save_consolidated = SaveConsolidatedMode.EVERY

        # 异步检查
        if self.is_async:
            assert _is_geq_torch_2_3(), "Async checkpoint requires torch >= 2.3.0"

    def build(
        self,
        *,
        dp_rank: int,
        tp_rank: int,
        pp_rank: int,
        moe_mesh: Optional["DeviceMesh"] = None,
    ) -> "Checkpointer":
        """注入运行时 rank / mesh 依赖，构造 Checkpointer 实例。

        与 optimizer / lr_scheduler 等 typed config 的 ``.build()`` 模式一致：
        YAML 层只提供静态字段，运行时依赖在此注入。
        """
        from .checkpointing import Checkpointer
        return Checkpointer(
            self,
            dp_rank=dp_rank,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            moe_mesh=moe_mesh,
        )


def _is_geq_torch_2_3() -> bool:
    major, minor = map(int, torch.__version__.split(".")[:2])
    return (major, minor) >= (2, 3)
    # NOTE: string-split comparison is fragile for pre-release versions (e.g. "2.3.0a1").
    # Safer alternative:
    #   from packaging import version
    #   return version.parse(torch.__version__) >= version.parse("2.3.0")
```

---

## 5. Checkpointer 核心类

> **调用位置**: 时序树 ④.7 — `CheckpointingConfig.build()` → `Checkpointer`

### 5.1 初始化

```python
# hyper_models/components/checkpoint/checkpointing.py

import json
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter, FileSystemReader
from torch.distributed.tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)

# 同包/跨包业务符号（第六轮 P0 修复：补全第五轮遗留的 import 半成品）
from .config import CheckpointingConfig, SerializationFormat, SaveConsolidatedMode
from .stateful_wrappers import ModelState, OptimizerState
from ._backports.hf_storage import _HuggingFaceStorageWriter, _HuggingFaceStorageReader
from .addons import ConsolidatedHFAddon, PeftAddon


class Checkpointer:
    """统一的 Checkpoint 管理器。

    生命周期：
    - 构造：在 recipe.setup() 中创建（step ④.7），用于训练过程中的 checkpoint 保存/恢复
    - 初始化加载：load_base_model() 是独立的自由函数，在 _build_model() 中调用（不依赖 Checkpointer 实例状态）
    - 运行时：save_model/save_optimizer 在训练循环中调用；load_model/load_optimizer 用于断点续训恢复
    - 销毁：训练结束时 close()（清理异步 future 和 process group）

    职责：
    - save_model / load_model: 模型权重的保存/恢复
    - save_optimizer / load_optimizer: 优化器状态的保存/恢复
    - 支持 DCP 切分 + HF safetensors 两种格式
    - 异步保存（torch >= 2.3.0）
    - StateDictAdapter 透明 key 转换
    """

    def __init__(
        self,
        config: CheckpointingConfig,
        *,
        dp_rank: int = 0,
        tp_rank: int = 0,
        pp_rank: int = 0,
        moe_mesh: Optional["DeviceMesh"] = None,
    ):
        self.config = config
        self.dp_rank = dp_rank
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self._moe_mesh = moe_mesh
        # 注（06 D-10 口径）：主 mesh 不含 EP 轴，expert mesh 由
        # apply_sharding_plan 期派生（sharding_applier._build_expert_mesh）
        # 且当前代码未导出——非 MoE 或 EP=1 时本参数为 None；MoE 模型的
        # consolidated 导出（adapter.to_hf(device_mesh=...)）需要派生
        # expert mesh 时，需先在 sharding 层暴露该 mesh 再在 Recipe.setup
        # 注入。这是已知的待落地缺口，不影响非 MoE 路径。

        # 异步上下文
        self._async_model_future = None
        self._async_optim_future = None
        self._async_stager = None

        if config.is_async:
            from torch.distributed.checkpoint.staging import DefaultStager
            self._async_stager = DefaultStager()

        # 保存时使用的 process group（默认 None 使用全局 pg）
        self._saving_pg: torch.distributed.ProcessGroup | None = None

        # Addons
        self._addons = []
        if config._serialization_format == SerializationFormat.SAFETENSORS and not config.is_peft:
            self._addons.append(ConsolidatedHFAddon())
        if config.is_peft:
            self._addons.append(PeftAddon())
```

### 5.2 保存模型

> **调用位置**: 时序树 ⑤ `save_checkpoint()` — `checkpointer.save_model()`

```python
def save_model(
    self,
    model: nn.Module | list[nn.Module],
    weights_path: str,
    peft_config=None,
    tokenizer=None,
    is_final_checkpoint: bool = False,
) -> None:
    """保存模型权重。

    流程：
    1. 创建 ModelState wrapper
    2. 获取 state dict → StateDictAdapter.to_hf()
    3. DCP 切分写入（每 rank 独立写 .distcp 文件）
    4. 可选：consolidated HF safetensors 导出
    """
    # ① 判断是否导出 consolidated
    should_consolidate = self._should_write_consolidated(is_final_checkpoint)

    # ② 创建 ModelState
    model_state = ModelState(
        model,
        is_peft=self.config.is_peft,
        skip_task_head_prefixes=self.config.skip_task_head_prefixes,
    )
    state_dict = model_state.state_dict()

    # ③ StateDictAdapter: 模型内部 key → HF key
    # PP 多 stage 时 adapter 从第一个 part 派生（与 load_model 对称）
    adapter_model = model[0] if isinstance(model, list) else model
    adapter = _get_state_dict_adapter(adapter_model)
    if adapter is not None:
        state_dict = adapter.to_hf(state_dict, device_mesh=self._moe_mesh)

    # ④ 材质化非连续视图（safetensors 要求）
    if self.config._serialization_format == SerializationFormat.SAFETENSORS:
        state_dict = _materialize_non_contiguous(state_dict)

    # ⑤ 构建文件索引 (fqn → shard file)
    if should_consolidate:
        fqn_to_file_index = get_fqn_to_file_index_mapping(state_dict)
        fqn_to_dtype = get_fqn_to_dtype_mapping(state_dict)
    else:
        fqn_to_file_index, fqn_to_dtype = None, None

    # ⑥ Addon pre-save（config.json, tokenizer, PEFT adapter）
    for addon in self._addons:
        addon.pre_save(
            weights_path, model, tokenizer, state_dict,
            fqn_to_file_index, peft_config,
        )

    # ⑦ 写入离线合并脚本
    if should_consolidate and not self.config.single_rank_consolidation:
        _write_consolidate_script(weights_path)

    # ⑧ 创建 StorageWriter
    if self.config._serialization_format == SerializationFormat.SAFETENSORS:
        writer = _HuggingFaceStorageWriter(
            weights_path, fqn_to_file_index, fqn_to_dtype,
        )
    else:
        writer = FileSystemWriter(weights_path)

    # ⑨ DCP 保存（model 的异步 future 写入 _async_model_future）
    self._do_save(state_dict, weights_path, writer, future_slot="_async_model_future")

    # ⑩ Addon post-save
    for addon in self._addons:
        addon.post_save(weights_path)

    # ⑪ Consolidated 导出（全 rank 并行合并）
    if should_consolidate and not self.config.single_rank_consolidation:
        consolidate_safetensors_files_on_every_rank(
            weights_path, fqn_to_file_index, fqn_to_dtype,
        )


def _should_write_consolidated(self, is_final: bool) -> bool:
    """判断是否需要导出 consolidated HF safetensors。

    ``save_consolidated="final"`` 仅在 ``is_final=True`` 时触发。
    调用侧契约（与 03 ``save_checkpoint`` 新签名一致）：03 的
    ``save_checkpoint(..., is_final_checkpoint=False)`` 将
    ``is_final_checkpoint`` 透传给 ``save_model``——训练结束的 final save
    传 ``True``，周期 save 使用默认 ``False``，final 模式因此只在训练
    结束时触发一次 consolidated 导出。
    """
    mode = self.config.save_consolidated
    if mode == SaveConsolidatedMode.EVERY:
        return True
    if mode == SaveConsolidatedMode.FINAL and is_final:
        return True
    return False


def _do_save(self, state_dict: dict, path: str, writer, *,
             future_slot: str = "_async_model_future") -> None:
    """执行 DCP 保存。

    dtensor_metadata 不进入 DCP SavePlan（torch 原生 ``SavePlan`` 不接受自定义
    字段），改写入 sidecar JSON（``{path}/.dtensor_metadata.json``），供外部
    审查工具与 ShardingPlan validate 模式消费。DCP 原生重分片由 DTensor 内部
    placements + mesh 元数据驱动，不依赖此 sidecar。

    async 契约（唯一一套）：``dcp.async_save``（torch >= 2.3，与
    ``CheckpointingConfig.is_async`` 的版本断言一致）返回
    ``torch.futures.Future``，本方法将其写入 ``future_slot`` 指定的实例
    属性——model 保存用 ``_async_model_future``、optimizer 保存用
    ``_async_optim_future``。若两处共用同一属性，save_optimizer 会覆盖
    model 的 future，``async_wait`` 将等不到 optimizer 保存完成。

    DTensorMetadata 定义见 §7。
    """
    # ① 提取 DTensor 元数据并写入 sidecar JSON（可观测性，不进 DCP SavePlan）
    # NOTE: sidecar 在 dcp.save 之前写入是设计意图——元数据不依赖保存成功；
    # 若后续 dcp.save 失败，sidecar 过期但无害（下次保存会覆写）。
    dtensor_metadata = _extract_dtensor_metadata(state_dict)
    os.makedirs(path, exist_ok=True)
    sidecar_path = os.path.join(path, ".dtensor_metadata.json")
    with open(sidecar_path, "w") as f:
        json.dump(
            {fqn: m.__dict__ for fqn, m in dtensor_metadata.items()},
            f,
            indent=2,
        )
        # NOTE: m.__dict__ serializes all dataclass fields including private/transient
        # ones. Prefer dataclasses.asdict(m) for a cleaner, public-fields-only dict:
        #   from dataclasses import asdict
        #   {fqn: asdict(m) for fqn, m in dtensor_metadata.items()}

    # ② DCP 保存（torch 原生 SavePlan 由 DCP 内部构造，用户只传 state_dict）
    if self.config.is_async and self._async_stager is not None:
        # 异步保存：dcp.async_save 返回 torch.futures.Future，写入 future_slot
        future = dcp.async_save(
            state_dict,
            checkpoint_id=path,
            storage_writer=writer,
            process_group=self._saving_pg,
            async_stager=self._async_stager,
        )
        setattr(self, future_slot, future)
    else:
        dcp.save(
            state_dict,
            checkpoint_id=path,
            storage_writer=writer,
        )
```

### 5.3 加载模型

> **调用位置**:
> - 时序树 ④.13 — `checkpointer.load_model()`（断点续训 resume，仅 DCP path）
> - 时序树 _build_model — `load_base_model()`（基座模型 init，路径 1 MoE merging / 路径 2 safetensors）
>
> init 路径已从 `Checkpointer.load_model` 迁入自由函数 `load_base_model`（见下方），
> `Checkpointer.load_model` 仅保留训练期 resume 语义。

```python
def load_model(
    self,
    model: nn.Module | list[nn.Module],
    model_path: str,
    allow_checkpoint_key_subset: bool = False,
) -> None:
    """从 DCP checkpoint 恢复模型权重（断点续训 resume）。

    仅负责训练期 resume 语义。基座模型初始化加载（from_pretrained 路径）
    由自由函数 ``load_base_model`` 负责（§5.3），支持 MoE tensor merging
    与 safetensors fast path 两条 init 路径。本方法不再处理 init。

    Args:
        model: 单个模型或 PP 多 stage 的 model_parts 列表。
        model_path: DCP checkpoint 目录。
        allow_checkpoint_key_subset: 加载时是否允许 checkpoint key 为模型 key 的子集。

    PP key 集合一致性（方案 a）：save 侧（03 save_checkpoint）保存的是单个
    ``self.model`` 的全模型 key 集合，而 load 侧传入 ``model_parts`` 列表、
    ``ModelState.state_dict()`` 逐 part 收集模板——合并模板与 checkpoint 的
    key 集合天然一致（同一全模型）。加载时 ``ModelState.load_state_dict``
    先对合并 key 集合做 strict 校验，再按 part 过滤子集、逐 part 以
    ``strict=False`` 的 ``set_model_state_dict`` 分发（每个 part 只取自己的
    key，合并 dict 是其超集）。选 (a) 而非 (b)（save 侧按 part 分文件存）
    的理由：保持磁盘格式与 PP 拓扑无关，单卡保存的 checkpoint 可直接用于
    PP 恢复（与跨配置重分片语义一致，见 §7.1）。
    """
    model_state = ModelState(
        model,
        is_peft=self.config.is_peft,
        is_init_step=False,
        skip_task_head_prefixes=self.config.skip_task_head_prefixes,
    )

    # ── DCP path（resume） ──
    # 获取当前模型的 state_dict 结构，作为 DCP load 的目标 shape/sharding 模板
    state_dict = model_state.state_dict()

    # adapter 方向约定（统一）：
    #   from_hf = HF key → 模型内部 key
    #   to_hf   = 模型内部 key → HF key
    # 保存时使用了 StateDictAdapter.to_hf() → checkpoint 中存储的是 HF key 格式。
    # pre-load：model_state.state_dict() 返回的是模型内部 key，需 to_hf（model→HF）
    #           匹配 checkpoint 内的 key，作为 DCP load 的目标模板。
    # PP 多 stage 时 adapter 从第一个 part 派生（所有 part 共享同一模型类型/key 映射）。
    adapter_model = model[0] if isinstance(model, list) else model
    adapter = _get_state_dict_adapter(adapter_model)
    if adapter is not None:
        state_dict = adapter.to_hf(state_dict, device_mesh=self._moe_mesh)

    if self.config._serialization_format == SerializationFormat.SAFETENSORS:
        reader = _HuggingFaceStorageReader(model_path)
    else:
        reader = FileSystemReader(model_path)

    dcp.load(
        state_dict,
        checkpoint_id=model_path,
        storage_reader=reader,
    )

    # post-load：dcp.load 后 state_dict 仍是 HF key 格式，要 load 进模型需
    #            from_hf（HF→model 内部 key）再 load_state_dict。
    if adapter is not None:
        state_dict = adapter.from_hf(state_dict)

    model_state.load_state_dict(state_dict, strict=not allow_checkpoint_key_subset)
```

```python
# hyper_models/components/checkpoint/loading.py

import json
import os

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)


def load_base_model(
    model: nn.Module,
    device: torch.device,
    path: str,
    adapter,
    mesh,
) -> None:
    """从 HF checkpoint 加载基座模型权重（from_pretrained 内部调用）。

    自由函数，不属于 Checkpointer 类。在 ``_build_model()`` 中调用，
    负责基座模型初始化的全部路径（原 ``Checkpointer.load_model`` 的 init 分支
    已迁入此处，``Checkpointer.load_model`` 仅保留训练期 resume 语义）。

    两条 init 路径：
      - 路径 1: MoE tensor merging（``requires_tensor_merging`` 为真时走 HF
        ``WeightConverter`` 合并分散的 expert 权重，零 NCCL）。
      - 路径 2: Safetensors fast path（每 rank 并行读，零 NCCL）。

    Args:
        model: 要加载权重的模型
        device: 目标设备
        path: safetensors checkpoint 路径
        adapter: StateDictAdapter（用于 HF key ↔ 模型内部 key 转换）
        mesh: DeviceMesh（用于 DTensor 分片）
    """
    # ── 路径 1: MoE tensor merging ──
    if requires_tensor_merging(_get_model_type(model)):
        # HF checkpoint 中 expert 权重分散存储，需用 WeightConverter 合并后再加载。
        # adapter 在此路径不介入（HF key 已与模型结构对齐）。
        state_dict = _convert_checkpoint_with_transformers(path, model.config)
        _load_full_state_dict_into_model(model, state_dict)
        _reinit_non_persistent_buffers(model)
        ensure_tied_lm_head(model)
        return

    # ── 路径 2: Safetensors fast path（零 NCCL） ──
    state_dict = _load_hf_checkpoint_preserving_dtype(path)

    # from_hf = HF key → 模型内部 key（HF safetensors 存储的是 HF key）
    if adapter is not None:
        state_dict = adapter.from_hf(state_dict, device_mesh=mesh)

    _load_full_state_dict_into_model(model, state_dict)

    # 重置非持久化 buffer（如 Gemma3 rotary embeddings）
    _reinit_non_persistent_buffers(model)

    # 确保 tied lm_head 正确
    ensure_tied_lm_head(model)


def _get_hf_safetensors_reference_path(root_dir: str, model_name: str | None) -> str:
    """解析 HF safetensors 的参考路径。"""
    import os
    from huggingface_hub.constants import HF_HUB_CACHE

    if model_name is not None:
        # 优先使用 root_dir/model_name
        candidate = os.path.join(root_dir, model_name)
        if os.path.exists(candidate):
            return candidate
    # fallback 到 root_dir
    if os.path.exists(root_dir):
        return root_dir
    # 尝试 HF cache
    if model_name is not None and HF_HUB_CACHE is not None:
        cache_path = os.path.join(HF_HUB_CACHE, "models--" + model_name.replace("/", "--"))
        if os.path.exists(cache_path):
            return cache_path
    return root_dir
```

### 5.4 每 rank 独立加载（零 NCCL）

> **调用位置**: 时序树 _build_model 路径2 — `load_base_model()` 内部（零 NCCL）

```python
# hyper_models/components/checkpoint/loading.py

def _load_full_state_dict_into_model(
    model: nn.Module,
    state_dict: dict,
) -> None:
    """将全量 state dict 加载到 DTensor 分片模型。

    关键：不设 broadcast_from_rank0=True！
    每个 rank 独立从全量 state dict 中切出自己的 DTensor shard。
    零 NCCL communication。

    实现利用 PyTorch 的 set_model_state_dict：
    - 参数是 DTensor → DTensor 自动切分（只取 local shard）
    - 参数不是 DTensor → 直接 load_state_dict(strict=False)
    """
    has_dtensor = any(
        isinstance(p, DTensor)
        for p in model.parameters()
    )

    if has_dtensor:
        options = StateDictOptions(
            full_state_dict=True,
            strict=False,
            # broadcast_from_rank0 不设置！每 rank 独立切分
        )
        set_model_state_dict(
            model,
            model_state_dict=state_dict,
            options=options,
        )
    else:
        # 非 DTensor 模型：标准加载
        model.load_state_dict(state_dict, strict=False)


def _load_hf_checkpoint_preserving_dtype(model_path: str) -> dict:
    """所有 rank 并行读 HF safetensors → 全量 CPU state dict。

    保留 checkpoint 原始 dtype（不转换到 torch_dtype），
    因为 FSDP2 的 MixedPrecisionPolicy 在 forward 时负责精度转换。
    """
    import json, os
    from safetensors.torch import load_file

    index_path = os.path.join(model_path, "model.safetensors.index.json")

    if os.path.exists(index_path):
        # 分片 checkpoint
        with open(index_path) as f:
            index = json.load(f)

        state_dict = {}
        shard_files = set(index["weight_map"].values())

        # 并行读所有 shard（每个 rank 独立读，无竞态）
        for shard_file in sorted(shard_files):
            shard_path = os.path.join(model_path, shard_file)
            shard_data = load_file(shard_path, device="cpu")
            state_dict.update(shard_data)

        return state_dict
    else:
        # 单文件
        return load_file(
            os.path.join(model_path, "model.safetensors"),
            device="cpu",
        )
```

### 5.5 异步保存

```python
def async_wait(self) -> None:
    """等待上一次异步保存完成（model + optimizer）。

    async 契约（唯一一套，旧 ``persist_completion`` 防御分支已删除）：
    ``dcp.async_save``（torch >= 2.3）返回 ``torch.futures.Future``，
    ``async_wait`` 对每个 future 调 ``.result()`` 阻塞至磁盘 upload 完成。
    旧 fork 的 ``AsyncSaveResponse.persist_completion`` 分支废弃——本仓库
    只以 torch 原生 async_save 为准（与 ``CheckpointingConfig.is_async``
    的 torch >= 2.3 版本断言一致）。
    """
    for future in (self._async_model_future, self._async_optim_future):
        if future is not None:
            future.result()
    self._async_model_future = None
    self._async_optim_future = None


def close(self) -> None:
    """训练结束时清理：等待异步 future、释放 process group（第六轮 P1 修复）。

    03 `run_train_validation_loop` 的 finally 块调用 `self.checkpointer.close()`，
    本方法此前缺失导致 AttributeError。
    """
    self.async_wait()
    if getattr(self, "_saving_pg", None) is not None:
        torch.distributed.destroy_process_group(self._saving_pg)
        self._saving_pg = None


def maybe_wait_for_staging(self) -> None:
    """等待 DCP staging（在 optimizer.step() 之前调用）。

    staging = 数据从 GPU 拷贝到 CPU pinned memory。
    upload = 数据从 CPU 写入磁盘——异步进行，不阻塞训练。
    ``DefaultStager.synchronize_staging()`` 是 staging 屏障：阻塞至上一轮 staging 拷贝完成，
    释放 pinned memory 以便复用，但**不**等待磁盘 upload 完成。
    """
    if self._async_stager is not None:
        self._async_stager.synchronize_staging()
```

---

## 6. ModelState / OptimizerState

> **调用位置**: 时序树 save/load 路径 — `ModelState(model)` / `OptimizerState(model, optimizer)`

```python
# hyper_models/components/checkpoint/stateful_wrappers.py

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)


class ModelState:
    """模型 StateDict wrapper — 处理 DCP、PEFT、tied weights。"""

    def __init__(
        self,
        model: nn.Module | list[nn.Module],
        is_peft: bool = False,
        is_init_step: bool = False,
        skip_task_head_prefixes: list[str] | None = None,
    ):
        self.model = model if isinstance(model, list) else [model]
        self.is_peft = is_peft
        self.is_init_step = is_init_step
        self.skip_task_head_prefixes = skip_task_head_prefixes or []

        # 检测 tied lm_head
        self.uses_tied_lm_head = _detect_tied_lm_head(model)
        self.lm_head_param_name = _get_lm_head_param_name(model) if self.uses_tied_lm_head else None

    def state_dict(self) -> dict:
        """获取 state dict。

        - init step: 返回 base model state dict（去除 lm_head 如果 tied）
        - PEFT: 只返回 LoRA keys
        - 正常: DTensor + full_state_dict
        """
        if self.is_init_step:
            return self._get_init_state_dict()

        if self.is_peft:
            return self._get_peft_state_dict()

        # 正常路径：保存 DTensor 的 local shard + 元数据，支持跨配置重分片
        # 不使用 full_state_dict=True —— 每个 rank 只保存自己的 shard，
        # DCP 在加载时根据 DTensor 内部元数据（placements + mesh）自动执行
        # all-gather + re-shard，原生支持跨配置重分片。
        options = StateDictOptions(
            full_state_dict=False,  # 保存 local shard，非全量
            cpu_offload=True,
            ignore_frozen_params=False,
        )
        # PP 多 stage 场景：遍历所有 model parts，汇总 state_dict
        state_dict = {}
        for part in self.model:
            part_state = get_model_state_dict(part, options=options)
            state_dict.update(part_state)

        # 移除 tied lm_head（避免重复保存）
        if self.uses_tied_lm_head and self.lm_head_param_name:
            state_dict.pop(self.lm_head_param_name, None)

        return state_dict

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """加载 state dict。"""
        if self.is_init_step:
            # Iterate all model parts (PP multi-stage: save every stage, not just model[0])
            for part in self.model:
                _set_base_model_state_dict(part, state_dict)
            for part in self.model:
                ensure_tied_lm_head(part)
        elif self.is_peft:
            self._load_peft_state_dict(state_dict)
        else:
            # PP key 集合修复（方案 a，理由见 §5.3 load_model 注释）：
            # save 侧（03 save_checkpoint）存的是 self.model 全模型 key，
            # load 侧传入 model_parts 列表。此处先合并校验、再按 part 过滤分发。
            if strict:
                # strict 语义在合并层面校验：state_dict 须覆盖所有 part 的 key
                missing: set[str] = set()
                for part in self.model:
                    missing |= set(part.state_dict().keys()) - set(state_dict.keys())
                if missing:
                    raise KeyError(
                        f"Missing keys in checkpoint state_dict: {sorted(missing)}"
                    )
            for part in self.model:
                # 材质化缺失的 tied lm_head
                if self.uses_tied_lm_head:
                    materialize_missing_tied_lm_head(part, state_dict)
                # 每个 part 只取自己的 key 子集——合并 state_dict 是各 part
                # key 的超集，逐 part 恒用 strict=False 加载子集
                part_keys = set(part.state_dict().keys())
                part_state = {k: v for k, v in state_dict.items() if k in part_keys}
                set_model_state_dict(
                    part,
                    model_state_dict=part_state,
                    options=StateDictOptions(strict=False),
                )
                ensure_tied_lm_head(part)


class OptimizerState:
    """优化器 StateDict wrapper（optimizer 接受单个或 list，与 03 §3.1 canonical 对齐）。

    03 canonical：``OptimizerConfig.build()`` 返回 ``list[torch.optim.Optimizer]``
    （PP 多 stage 时每个 model part 一个 optimizer，见 03 §9.3），
    ``__state_tracked`` 注册的 optimizer 即该 list。本 wrapper 内部统一归一为
    list；``get_optimizer_state_dict`` / ``set_optimizer_state_dict`` 原生支持
    ``(model, optimizers)`` 的 list 传参（torch >= 2.2）。
    """

    def __init__(self, model: nn.Module | list[nn.Module],
                 optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer],
                 scheduler: torch.optim.lr_scheduler.LRScheduler | None = None):
        self.models = model if isinstance(model, list) else [model]
        self.optimizers = optimizer if isinstance(optimizer, list) else [optimizer]
        self.scheduler = scheduler

    def state_dict(self) -> dict:
        result = {}

        # FSDP2 使用 DTensor 优化器状态，不需要 flatten
        # 按 part_{i} 分桶保存（与 ModelState 对称）：
        # - 单模型（len(models) == 1）：直接把 self.optimizers 作为 list 传给
        #   get_optimizer_state_dict（原生支持 list，单模型多 optimizer 也成立）
        # - PP 多 stage：models 与 optimizers 等长（03 §9.3：每 part 一个
        #   optimizer），逐 (part, optimizer) 配对调用
        optim_bucket = {}
        if len(self.models) == 1:
            optim_bucket["part_0"] = get_optimizer_state_dict(
                self.models[0], self.optimizers,
                options=StateDictOptions(flatten_optimizer_state_dict=False),
            )
        else:
            assert len(self.models) == len(self.optimizers), (
                "PP 多 stage 时 models 与 optimizers 必须等长（每 part 一个 optimizer）"
            )
            for i, (m, o) in enumerate(zip(self.models, self.optimizers)):
                optim_bucket[f"part_{i}"] = get_optimizer_state_dict(
                    m, o,
                    options=StateDictOptions(flatten_optimizer_state_dict=False),
                )
        result["optim"] = optim_bucket
        if self.scheduler is not None:
            result["sched"] = self.scheduler.state_dict()
        return result

    def load_state_dict(self, state_dict: dict) -> None:
        # 与 state_dict() 对称：单模型直接传 list；PP 逐 (part, optimizer) 配对
        if len(self.models) == 1:
            set_optimizer_state_dict(
                self.models[0], self.optimizers,
                optim_state_dict=state_dict["optim"]["part_0"],
            )
        else:
            for i, (m, o) in enumerate(zip(self.models, self.optimizers)):
                set_optimizer_state_dict(
                    m, o,
                    optim_state_dict=state_dict["optim"][f"part_{i}"],
                )
        if self.scheduler is not None and "sched" in state_dict:
            self.scheduler.load_state_dict(state_dict["sched"])
```

---

## 7. DCP + DTensor 元数据扩展

> **调用位置**: 时序树 dcp.save() — 可观测性层

### 7.1 设计定位

PyTorch DCP（`torch.distributed.checkpoint`）在保存 DTensor 时**已经原生记录** placements、mesh 信息等元数据。hyper_parallel 使用 `full_state_dict=False` 模式保存 local shard——每个 rank 只写自己的 shard 数据，DCP 同时记录 DTensor 的 placements 和 mesh_dim_names 作为内部元数据。加载时 DCP 自动检测 placements 不匹配并执行 all-gather + re-shard，从而**原生支持跨配置重分片**。

hyper_parallel 的 `DTensorMetadata` 是一个**附加的可观测性层**，服务于两个非功能需求：

1. **调试/审计**：外部工具可以解析 checkpoint 内的分片布局，无需加载整个模型
2. **ShardingPlan 验证**：validate 模式下，将保存时的实际 placements 与 ShardingPlan 的预期值进行 diff

```python
# core/distributed_checkpoint/planner.py (修改)

from dataclasses import dataclass
from torch.distributed.tensor import DTensor

# NOTE: 不再自定义 SavePlan——torch 原生 ``torch.distributed.checkpoint.SavePlan``
# 由 DCP 内部构造，不接受自定义字段。DTensor 元数据改写入 sidecar JSON
# （见 §5.2 ``_do_save``），与 torch DCP 兼容。``DTensorMetadata`` 仍保留为
# 可观测性数据结构，由 ``_extract_dtensor_metadata`` 产出并序列化到 sidecar。


@dataclass
class DTensorMetadata:
    """单个参数的 DTensor 元数据（可观测性，非核心机制）。

    DCP 原生重分片不需要此结构——DCP 内部已包含 placements + mesh 信息。
    此结构提供人类可读的序列化格式，用于：
    - checkpoint 审查工具（无需加载模型即可查看分片布局）
    - ShardingPlan validate 模式（预期 vs 实际 placements diff）
    """
    fqn: str                        # 参数 FQN
    global_shape: tuple[int, ...]   # 未分片的全局形状
    placements: list[str]           # placement 序列化 (如 ["Shard(0)", "Replicate()"])
    mesh_dim_names: tuple[str, ...] # mesh 维度名
    mesh_shape: tuple[int, ...]     # mesh 形状


def _extract_dtensor_metadata(state_dict: dict) -> dict[str, DTensorMetadata]:
    """从 state dict 中提取 DTensor 元数据（仅用于可观测性）。"""
    metadata = {}
    for fqn, tensor in state_dict.items():
        if isinstance(tensor, DTensor):
            metadata[fqn] = DTensorMetadata(
                fqn=fqn,
                global_shape=tuple(tensor.size()),  # DTensor 的 .size() 与 .shape 均返回全局形状；本地分片形状为 tensor.to_local().shape
                placements=[str(p) for p in tensor.placements],
                mesh_dim_names=tensor.device_mesh.mesh_dim_names,
                mesh_shape=tuple(tensor.device_mesh.mesh.shape),
            )
    return metadata
```

**跨配置重分片由 PyTorch DCP 原生处理**——使用 `full_state_dict=False` 保存 local shard + DTensor 元数据，DCP 加载时自动检测 placements 不匹配并执行 all-gather + re-shard，无需在 `DTensorMetadata` 中重复实现此逻辑。如果未来发现 DCP 原生重分片有性能或功能局限，可以在此基础上扩展。`StandardLoadPlanner` 的 `create_load_plan` 也无需自定义实现，DCP 的默认 planner 已处理此逻辑。

**production 参数形态与 tp_grad_info 衔接**：保存时 `ModelState.state_dict()` 走 `full_state_dict=False`，此时 FSDP2 内部 `sharded_param` 暴露的 DTensor 其 placements 形如 `[Shard(DP), tp_placement]`（DP 维由 `fully_shard` 管理，TP 维来自 build 期 ShardingPlan）。这与 05/06 的 build 期解包 + `tp_grad_info` 机制衔接：build 期 `_local_params_context` 已将 `DTensor[TP]` 解包为 plain local tensor，`fully_shard` 随后以 `LOCAL_PARAM + tp_grad_info` 管理梯度，运行期参数对用户代码呈现为 plain tensor；但 FSDP2 内部仍以 `[Shard(DP), tp_placement]` 的 DTensor 记录分片语义，保存时即被 DCP 元数据捕获。因此 `DTensorMetadata` 中观察到的 TP placement 来源是 ShardingPlan（经 `tp_grad_info` 注入），**而非**从 DTensor placement 现场推导——这与 05 §6.7 canonical 一致（tp_grad_info 从 ShardingPlan 读取，具体见 05 §6.7.1 ``build_tp_grad_info``）。

---

### 7.2 辅助函数签名

```python
# ── Checkpointer 方法（save/load optimizer） ──
def save_optimizer(
    self,
    model: nn.Module | list[nn.Module],
    optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer],
    path: str,
    scheduler=None,
) -> None:
    """保存优化器状态（含 LR scheduler，若传入非 None 则绑定）。

    与 save_model 对称：组装 OptimizerState → DCP 切分写入。
    ``optimizer`` 接受单个或 list——与 03 §3.1 canonical 对齐
    （``OptimizerConfig.build()`` 返回 ``list[torch.optim.Optimizer]``，
    ``__state_tracked`` 注册的 optimizer 即该 list）。
    """
    opt_state = OptimizerState(model, optimizer, scheduler=scheduler)
    state_dict = opt_state.state_dict()
    if self.config._serialization_format == SerializationFormat.SAFETENSORS:
        writer = _HuggingFaceStorageWriter(path, None, None)
    else:
        writer = FileSystemWriter(path)
    self._do_save(state_dict, path, writer, future_slot="_async_optim_future")


def load_optimizer(
    self,
    model: nn.Module | list[nn.Module],
    optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer],
    path: str,
) -> None:
    """从 DCP checkpoint 恢复优化器状态（断点续训 resume）。

    ``optimizer`` 签名与 save_optimizer 一致（list 与 03 §3.1 canonical 对齐）。
    """
    opt_state = OptimizerState(model, optimizer, scheduler=None)
    state_dict = opt_state.state_dict()
    if self.config._serialization_format == SerializationFormat.SAFETENSORS:
        reader = _HuggingFaceStorageReader(path)
    else:
        reader = FileSystemReader(path)
    dcp.load(state_dict, checkpoint_id=path, storage_reader=reader)
    opt_state.load_state_dict(state_dict)

# ── Recipe 侧故障恢复辅助（§8 load_checkpoint 调用） ──
def _validate_checkpoint_compatibility(self, restore_from: str) -> None:
    """校验 checkpoint 与当前运行环境兼容性（DP size / TP size / PP size 等）。

    从 checkpoint 目录的 extra_state.json 或 .dtensor_metadata.json 读取
    保存时的并行度配置，与当前运行环境的 mesh 维度对比。若 DP/TP/PP size
    或 DTensor placements 不匹配，抛出 RuntimeError（对断点续训而言意味着
    必须用相同拓扑恢复），或记录 warning（对部分兼容场景）。
    """
    ...

def _resolve_latest_symlink(checkpoint_dir: str) -> str | None:
    """读取 LATEST symlink 指向的最新 checkpoint 目录，不存在则返回 None。

    03 §7.1 ``_update_latest_symlink`` 写入的是**相对路径**
    （``os.path.relpath(path, checkpoint_dir)``，即 checkpoint 子目录名），
    消费端必须拼回 ``checkpoint_dir`` 再判 exists——直接对 readlink 结果
    调 ``os.path.exists`` 会依赖 CWD，CWD ≠ checkpoint_dir 时误判不存在。
    """
    import os
    symlink = os.path.join(checkpoint_dir, "LATEST")
    if os.path.islink(symlink):
        target = os.path.join(checkpoint_dir, os.readlink(symlink))
        if os.path.exists(target):
            return target
        return None
    return _maybe_load_latest_marker(checkpoint_dir)

def _maybe_load_latest_marker(checkpoint_dir: str) -> str | None:
    """无 symlink 时尝试读取 LATEST marker 文件（兼容无符号链接的 FS）。

    读取 ``{checkpoint_dir}/LATEST`` 纯文本文件（非 symlink），将其中
    记录的最后一行 checkpoint 子目录名拼回 ``checkpoint_dir`` 后返回
    完整路径（与 symlink 分支一致，不依赖 CWD）。不存在或无内容时返回 None。
    """
    import os
    marker = os.path.join(checkpoint_dir, "LATEST")
    if os.path.isfile(marker) and not os.path.islink(marker):
        with open(marker) as f:
            lines = f.read().strip().splitlines()
            if lines:
                return os.path.join(checkpoint_dir, lines[-1])
    return None

# ── StateDictAdapter 获取（canonical：01 §2.14） ──
def _get_state_dict_adapter(model: nn.Module):
    """从模型读取 ``_state_dict_adapter`` 属性（canonical：01 §2.14）。

    - 模型经 01 §11 ``HFCheckpointingMixin`` 持有 ``_state_dict_adapter``
      实例（注册期绑定，如 ``Qwen3_5DenseStateDictAdapter()``）→ 返回该实例。
    - 无该属性或属性为 None（无 HF key 映射需求）→ 返回 None
      （checkpoint key 与模型内部 key 一致）。

    注意：01 §10.1 的 ``StateDictAdapter`` ABC 上**不存在**
    ``from_model_type()`` 类方法，旧版按 ``model_type`` 派发的实现已删除。
    本函数即 01 §8.3 从 ``components.checkpoint.checkpointing`` import 的
    实现——checkpointing 模块提供的是"读取模型属性 + fallback None"的
    薄封装，不做类型派发。

    返回的 adapter 提供 ``to_hf(state_dict)``（模型内部 key → HF key）和
    ``from_hf(state_dict)``（HF key → 模型内部 key）两个方向。
    """
    return getattr(model, "_state_dict_adapter", None)

# ── safetensors 序列化辅助 ──
def _materialize_non_contiguous(state_dict: dict) -> dict:
    """将非连续 tensor 物料化为连续视图（safetensors 写入要求）。

    safetensors 只能写入内存连续的 tensor。对于因 view / transpose / slice
    等操作产生的非连续 tensor，需先调用 ``.contiguous()`` 创建连续副本。

    返回新 dict（不修改原始 state_dict），其中非连续的 DTensor / Tensor
    被替换为连续版本；已连续的 tensor 保留原位。
    """
    result = {}
    for fqn, tensor in state_dict.items():
        if hasattr(tensor, 'is_contiguous') and not tensor.is_contiguous():
            result[fqn] = tensor.contiguous()
        else:
            result[fqn] = tensor
    return result
def get_fqn_to_file_index_mapping(state_dict: dict) -> dict:
    """构建 fqn → shard file index 映射，供 HF index.json 使用。

    遍历 state_dict，按 tensor 字节大小等权分配各 shard（目标每 shard
    约 5GB，即 ``max_shard_size="5GB"``），返回 ``{fqn: file_index}``。
    最终写入 ``model.safetensors.index.json`` 的 ``weight_map`` 字段。
    """
    ...
def get_fqn_to_dtype_mapping(state_dict: dict) -> dict:
    """构建 fqn → dtype 字符串映射，供 HF index.json 使用。

    遍历 state_dict，将每个 fqn 的 dtype 记录为字符串
    （如 ``"torch.bfloat16"``），与 ``get_fqn_to_file_index_mapping``
    配合写入 ``model.safetensors.index.json`` 的 ``weight_map`` 元数据。
    """
    ...
def _write_consolidate_script(weights_path: str) -> None:
    """写入离线合并脚本 consolidate.sh，供单 rank 合并使用。

    生成 shell 脚本，内容为 python 一行命令调用 ``consolidate_safetensors``
    读取各 dp_rank_N/*.distcp shard 并合并写入 consolidated/ 目录。
    脚本写入 ``{weights_path}/consolidate.sh``。
    """
    import os
    script = os.path.join(weights_path, "consolidate.sh")
    with open(script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Consolidated HF safetensors merge script\n")
        f.write(f"python -m hyper_models.components.checkpoint.consolidate \\\n")
        f.write(f"  --input-dir {weights_path} \\\n")
        f.write(f"  --output-dir {weights_path}/consolidated\n")
    os.chmod(script, 0o755)

def consolidate_safetensors_files_on_every_rank(
    weights_path: str, fqn_to_file_index: dict, fqn_to_dtype: dict,
) -> None:
    """每 rank 并行合并各自 shard → 最终 .safetensors 文件。

    每个 rank 根据 ``fqn_to_file_index`` 确定自己要负责的 shard index，
    从 ``dp_rank_{i}/*.distcp`` 读取对应 fqn 的权重，写入
    ``consolidated/model-{shard_idx+1:05d}-of-{total:05d}.safetensors``。
    所有 rank 并行写入不同 shard 文件，无写竞态。最后写入
    ``model.safetensors.index.json``。
    """
    ...

# ── 基座模型加载辅助（load_base_model 路径 1/2） ──
def requires_tensor_merging(model_type: str) -> bool:
    """判断模型类型是否需要 MoE expert 权重合并（如 DeepSeek/Mixtral 分散存储）。

    返回 True 的模型类型：DeepSeek-V2/V3、Mixtral、Qwen2-MoE 等——
    它们的 HF checkpoint 将 expert 权重分散在多个 safetensors shard 中，
    需用 HF WeightConverter 合并为单张权重 tensor 后加载。
    """
    return model_type in (
        "deepseek_v2", "deepseek_v3",
        "mixtral", "qwen2_moe",
    )

def _get_model_type(model: nn.Module) -> str:
    """从模型 config.model_type 提取字符串标识（小写、去连字符）。"""
    if hasattr(model, "config") and hasattr(model.config, "model_type"):
        return model.config.model_type.replace("-", "_").lower()
    return type(model).__name__.lower()

def _convert_checkpoint_with_transformers(model_path: str, config) -> dict:
    """使用 HF WeightConverter 合并分散的 expert 权重 → 全量 state dict。

    加载 HF checkpoint 的 safetensors 文件，通过 ``WeightConverter`` 将
    分散在多个 shard 中的 MoE expert 权重合并为单张完整 tensor。
    返回可直接加载的 flat state dict（key 为模型内部 FQN）。
    """
    from hyper_models.components.checkpoint.conversion_mapping import WeightConverter
    converter = WeightConverter(model_path, config)
    return converter.convert()

def _reinit_non_persistent_buffers(model: nn.Module) -> None:
    """重置非持久化 buffer（如 Gemma3 rotary embeddings）。

    遍历模型的非持久化 buffer，调用其 reset_parameters()（如有）或调用模型
    自身的初始化方法重新填充值。RoPE 等 buffer 在加载 state dict 后需要重新
    计算（因为其值取决于 max_seq_len 等运行时参数，而非从 checkpoint 恢复）。

    注意：PyTorch 的 nn.Buffer 没有 ``_persistent`` 属性，且 ``named_buffers()``
    默认会同时返回持久化与非持久化 buffer。正确判定方式是查每个 module 的
    ``_non_persistent_buffers_mapping``（PyTorch 在 register_buffer(persistent=False)
    时维护的私有映射）。
    """
    for module in model.modules():
        non_persistent = getattr(module, "_non_persistent_buffers_mapping", {})
        for name, buf in non_persistent.items():
            if hasattr(buf, "reset_parameters"):
                buf.reset_parameters()
def _load_full_state_dict_into_model(model: nn.Module, state_dict: dict) -> None:
    """将全量 state dict 加载到 DTensor 分片模型（零 NCCL）。完整实现见 §5.4。"""
    ...
def _load_hf_checkpoint_preserving_dtype(model_path: str) -> dict:
    """所有 rank 并行读 HF safetensors → 全量 CPU state dict（保留原始 dtype）。完整实现见 §5.4。"""
    ...
def _set_base_model_state_dict(model: nn.Module, state_dict: dict) -> None:
    """init step 下将 base model state dict 设入模型（ModelState.load_state_dict 调用）。

    使用 ``set_model_state_dict``（DTensor 模型）或 ``model.load_state_dict``
    （非 DTensor 模型），strict=False 以容忍模型新增/移除的参数。
    """
    from torch.distributed.tensor import DTensor
    from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
    has_dtensor = any(isinstance(p, DTensor) for p in model.parameters())
    if has_dtensor:
        set_model_state_dict(model, model_state_dict=state_dict,
                            options=StateDictOptions(full_state_dict=True, strict=False))
    else:
        model.load_state_dict(state_dict, strict=False)

# ── tied weights 辅助 ──
def _detect_tied_lm_head(model) -> bool:
    """检测模型是否使用 tied lm_head（embed_tokens 与 lm_head 共享权重）。

    通过检查模型的 ``_tied_weights_keys`` 属性（部分 HF 模型设置）或
    比较 embed_tokens.weight 与 lm_head.weight 是否指向同一存储。
    返回 True 若存在 tied lm_head。
    """
    ...

def _get_lm_head_param_name(model) -> str:
    """返回 lm_head 参数在 state dict 中的 key 前缀（如 ``"lm_head"``）。

    从模型结构中推断——通常为最后一层 decoder/transformer 输出投影的 key。
    """
    ...

def ensure_tied_lm_head(model) -> None:
    """确保 tied lm_head 正确绑定：若模型声明了 tied weights 但实际未绑定，
    将 lm_head.weight 指向 embed_tokens.weight（共享存储，不复制数据）。

    注意：HF 的 ``_tied_weights_keys`` 是 ``List[str]``（每个元素是单个被 tie 的
    参数名，如 ``"lm_head.weight"``），**不是** pair 列表——不能按 ``len==2`` 解包。
    绑定关系的权威来源是模型自身的 ``tie_weights()``（HF 标准）与
    ``config.tie_word_embeddings``；这里在 load 后补一次绑定，保证 lm_head.weight
    与 embed_tokens.weight 共享 data_ptr（分片场景下二者须为同一 DTensor/local tensor）。
    """
    cfg = getattr(model, "config", None)
    if not getattr(cfg, "tie_word_embeddings", False):
        return

    # 优先走 HF 标准 tie_weights()（处理 _tied_weights_keys 的标准语义）
    if hasattr(model, "tie_weights"):
        model.tie_weights()
        return

    # 回退：显式将 lm_head.weight 绑定到 embed_tokens.weight
    def _get_param(fqn_candidates):
        for fqn in fqn_candidates:
            try:
                return model.get_parameter(fqn)
            except AttributeError:
                continue
        return None

    embed = _get_param(("embed_tokens.weight", "model.embed_tokens.weight"))
    lm_head = _get_param(("lm_head.weight",))
    if embed is not None and lm_head is not None and embed.data_ptr() != lm_head.data_ptr():
        # 共享同一底层 tensor（不复制数据）；分片场景下二者须为同一 DTensor/local
        lm_head.data = embed.data
        # 保持 grad 指向同一 AccumulateGrad 节点（tied 参数梯度在 embed 上累积）

def materialize_missing_tied_lm_head(model, state_dict: dict | None = None) -> None:
    """从 state dict 或 embed_tokens 中恢复缺失的 lm_head 参数。

    若 state dict 中不存在 lm_head.weight 但模型声明了 tied lm_head，
    从 embed_tokens.weight 创建共享存储的 lm_head 参数。
    """
    ...
```

### 7.3 MoE stacked 参数 key 映射（05 D-09 衔接）

> 对应 05 §6.4.7（D-09：HF 原生 MoE 的 EP 直通）。v1 的 key 转换逻辑归
> checkpoint 层，05 的 apply 不感知。

D-09 在 apply 期把 HF per-expert 权重 stack 成 3D 参数后，**内存中的参数 key
与 HF checkpoint 的 key 结构不再一致**：

```
HF checkpoint（磁盘）          训练内存（D-09 堆叠后）
─────────────────────          ──────────────────────
mlp.experts.0.gate_proj.weight  mlp.experts.gate_proj   [E, I, H]（holder Parameter，
mlp.experts.1.gate_proj.weight  mlp.experts.up_proj      无序号段、无 .weight 后缀，
...                             mlp.experts.down_proj    EP Shard(0) + TP D-08 分片）
mlp.experts.0.down_proj.weight
...
```

三个方向的约定：

| 方向 | 路径 | 转换 |
|------|------|------|
| **HF → 训练（init 加载）** | `load_base_model` 在 `apply_sharding_plan` **之前**执行（01 §7 时序） | **零转换**——per-expert 权重原样加载进模型，Phase A 的 `_stack_moe_experts` 在堆叠时自行 concat；推荐路径，WeightConverter 无需感知 stacked 格式 |
| **训练 → HF 导出（save）** | `Adapter.to_hf`（§5.2 保存 5 阶段的第 2 步） | **unstack**：stacked 参数按 E 维 split 回 per-expert key（补序号段与 `.weight` 后缀）；与现有 MoE tensor merging（§5.3 路径 1）互为逆操作，复用同一 key 映射表 |
| **DCP resume（自格式）** | `full_state_dict=False` local shard 保存 | **零转换**——stacked key 直接入盘，DTensor 元数据记录 `{EP: Shard(0), TP: ...}`，加载时 DCP 原生 re-shard（§7.1）；模型侧须已先经 apply 堆叠出同名参数（先 apply 后 resume 的固定时序） |

注意：unstack 需要知道 expert 数 E 与 proj 命名映射（gate_proj/up_proj/down_proj
↔ w1/w2/w3 两套命名），映射表按 arch 注册（与 05 `MOE_ROUTER_ADAPTERS` 同
机制），v1 覆盖 DeepSeek/Qwen3-MoE/GLM；未注册 arch 保存时 fail-fast 报错
（不静默写出无法回读的格式）。

---

## 8. 故障恢复集成

> **调用位置**: 时序树 ④.13 — `load_checkpoint()` 完整流程

```python
# recipes/base_recipe.py 中 load_checkpoint 的完整流程
# 遍历 __state_tracked 注册表，按种类分类加载，而非硬编码 7 步流程

def load_checkpoint(self, restore_from: str | None) -> None:
    """从 checkpoint 恢复所有组件。

    restore_from 解析：
    - None → 跳过恢复
    - "LATEST" → 读取 LATEST symlink
    - "epoch_0_step_100" → 直接使用路径
    """
    if restore_from is None:
        return

    if restore_from == "LATEST":
        restore_from = _resolve_latest_symlink(self.cfg.checkpoint.checkpoint_dir)
        if restore_from is None:
            logger.info("No LATEST checkpoint found, starting from scratch.")
            return

    if not os.path.exists(restore_from):
        logger.warning("Checkpoint %s not found, starting from scratch.", restore_from)
        return

    logger.info("Loading checkpoint from %s", restore_from)

    # ① 兼容性检查
    self._validate_checkpoint_compatibility(restore_from)

    # ② 遍历 __state_tracked 注册表，按 (name, kind) 对称解包加载（与 03 save 侧对称，
    #    kind 来自注册时绑定，不再用 _get_state_kind 重推导）
    for name, kind in sorted(self.__state_tracked):
        path = self._state_path(restore_from, name, kind)
        if not os.path.exists(path):
            continue
        self._load_state_by_kind(name, kind, path)


def _load_state_by_kind(self, name: str, kind: str, path: str) -> None:
    """按 state 种类分发加载（kind 来自 __state_tracked 注册绑定，不重推导）。"""
    obj = getattr(self, name)
    if kind == "model":
        # PP 多 stage：传 model_parts 列表。save 侧存的是 self.model 全模型
        # key（03 save_checkpoint 传单个模型），load 侧 ModelState 合并校验后
        # 按 part 过滤子集分发（方案 a，见 §5.3 load_model 说明）
        self.checkpointer.load_model(self.model_parts, model_path=path)
    elif kind == "optimizer":
        # obj 为 list[Optimizer]（03 §3.1 canonical）；传 model_parts 列表，
        # OptimizerState 内部逐 (part, optimizer) 配对恢复
        self.checkpointer.load_optimizer(self.model_parts, obj, path)
    elif kind == "lr_scheduler":
        # lr_scheduler build 返回 list[OptimizerParamScheduler]（见 03 §9.6），
        # save 侧聚合为 {f"sch_{i}": state_dict} 存入 scheduler.pt。
        # 此处遍历 list 逐个 load_state_dict，与 save 侧对称。
        state = torch.load(path, weights_only=False)
        schedulers = obj if isinstance(obj, list) else [obj]
        for i, sch in enumerate(schedulers):
            sch.load_state_dict(state[f"sch_{i}"])
    elif kind == "train_state":
        with open(path) as f:
            extra = json.load(f)
        obj.load_state_dict(extra)
    elif kind in ("rng", "dataloader"):
        state = torch.load(path, weights_only=False)
        obj.load_state_dict(state)


def _state_path(self, root: str, name: str, kind: str) -> str:
    """根据 state 种类和 name 计算 checkpoint 子路径。

    本方法定义在 BaseRecipe 上，self 即 Recipe 实例。
    """
    if kind in ("model", "optimizer"):
        return f"{root}/{kind}"
    if kind in ("rng", "dataloader"):
        return f"{root}/{kind}/{kind}_dp_rank_{self._get_dp_rank()}.pt"
    if kind == "train_state":
        return f"{root}/extra_state.json"
    if kind == "lr_scheduler":
        # 与 03 save 侧一致：写 `{path}/scheduler.pt`（见 03 save_checkpoint）
        return f"{root}/scheduler.pt"
    return f"{root}/{name}.pt"
```

---

## 9. RecipeConfig 桥接

> **调用位置**: 时序树 ④.7 — `RecipeConfig.checkpoint` cached_property（`_target_` → 类型化 Config → `.build()`）

AutoModel 对 checkpoint 使用与 optimizer 相同的**两层 typed config 模式**：

1. **RecipeConfig.checkpoint**：从 YAML 提取 kwargs（丢弃 `_target_`，如果有的话）→ 直接构造 `CheckpointingConfig`
2. **checkpoint_config.build(dp_rank=..., ...)**：注入运行时依赖，创建 `Checkpointer`

**canonical 实现归 01 §3.3**（`RecipeConfig.checkpoint` cached_property，含
`model_repo_id` / `model_cache_dir` / `is_peft` 等模型派生字段的注入），
本文档不再重复完整实现，仅保留 04 特有的 checkpoint 配置说明：

- `CheckpointingConfig` 是**固定类型**（不走 `_target_`），cached_property 直接
  用 `_section_kwargs()` 提取 YAML checkpoint 段字段后构造；
- `restore_from` 由 Recipe 单独解析（`load_checkpoint(restore_from)`），
  不传入 `CheckpointingConfig`；
- 模型派生字段（`model_repo_id` / `model_cache_dir` / `is_peft`）在
  cached_property 中注入，YAML 显式设置的值优先（详见 01 §3.3）。

### 使用方式

```python
# Recipe.setup() 中
checkpoint_config = self.cfg.checkpoint  # → CheckpointingConfig 实例（已类型校验）
self.checkpointer = checkpoint_config.build(
    dp_rank=self._get_dp_rank(),
    tp_rank=self._get_tp_rank(),
    pp_rank=self._get_pp_rank(),
    moe_mesh=getattr(self.mesh, "moe_mesh", None),
    # 06 D-10 口径：MeshContext 无 moe_mesh 字段（主 mesh 不含 EP 轴，
    # expert mesh 由 apply_sharding_plan 期派生且当前未导出），getattr 恒为
    # None；MoE consolidated 导出需要派生 expert mesh 时，需由 sharding 层
    # 暴露后在此注入（已知缺口，见 §5.1 Checkpointer.__init__ 的 moe_mesh 注）。
)
```

## 10. 配置示例

```yaml
recipe: FinetuneRecipe

# Checkpoint 配置（typed —— RecipeConfig 直接构造 CheckpointingConfig）
checkpoint:
  checkpoint_dir: outputs/qwen35_08b
  model_save_format: safetensors       # "safetensors" | "torch_save"
  save_consolidated: final             # "false" | "final" | "every"
  is_async: true                       # torch >= 2.3.0
  best_metric_key: loss
  restore_from: LATEST                 # 由 Recipe 单独解析，不传给 CheckpointingConfig
```

**与 AutoModel 的 `_target_` 使用对比总表**：

| 组件 | 路径 | YAML `_target_` | Recipe 调用 |
|------|------|----------|-----------|
| Model | **untyped** `.instantiate()` | `_target_: ...from_pretrained` | `cfg.model.instantiate(distributed_setup=...)` |
| Dataset | **untyped** `.instantiate()` | `_target_: datasets.load_dataset` | `cfg.dataset.instantiate(tokenizer=...)` |
| DataLoader | **untyped** `.instantiate()` | `_target_: ...StatefulDataLoader` | `cfg.dataloader.instantiate(dataset=...)` |
| Tokenizer | **untyped** `.instantiate()` | `_target_: AutoTokenizer.from_pretrained` | `cfg.dataset.tokenizer.instantiate()` |
| Collate | **untyped** `.instantiate()` | `_target_: ...default_collater` | `collate_cfg.instantiate(batch=batch)` |
| PEFT | **untyped** `.instantiate()` | `_target_: ...PeftConfig` | `cfg.peft.instantiate()` |
| **Optimizer** | **typed** `.build()` | `_target_: torch.optim.AdamW` | `cfg.optimizer.build(model, device_mesh=...)` |
| **LR Scheduler** | **typed** `.build()` | 无（固定类型 `LRSchedulerConfig`） | `cfg.lr_scheduler.build(optimizer, step_scheduler)` |
| **StepScheduler** | **typed** `.build()` | 无（固定类型 `StepSchedulerConfig`） | `cfg.step_scheduler.build(dataloader, dp_size, local_bs)` |
| **Loss** | **typed** `.build()` | `_target_: ...MaskedCrossEntropy` | `cfg.loss_fn.build()` |
| **Checkpoint** | **typed** `.build()` | 无（固定类型 `CheckpointingConfig`） | `cfg.checkpoint.build(dp_rank, tp_rank, ...)` |
| WandB/MLflow | **typed** `.build()` | 无（固定类型） | `cfg.wandb.build(run_config=...)` |
| RNG | **直接构造** | 无（`seed` 字段） | `StatefulRNG(seed=cfg.get("seed", 42), ranked=True)` |
