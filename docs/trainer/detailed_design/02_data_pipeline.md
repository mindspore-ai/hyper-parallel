# Hyper-Parallel 数据管道详细设计

> 参考实现：AutoModel `nemo_automodel/components/datasets/`（~30 个数据集类，完整 loader/collate/packing 体系）
> VeOmni `veomni/data/`（~15 个核心模块，含动态 batching 和 BackgroundPrefetcher）
> 上下文设计：[dual_mode_dtensor_parallel_strategy.md](../dual_mode_dtensor_parallel_strategy.md)

---

## 1. 模块职责

提供统一的数据管道，支持 **HF datasets**、**Megatron 二进制格式**、**对话格式（ChatDataset）**、**多源加权采样** 四种数据源，通过 `_target_` 声明式配置（配置容器为强类型 dataclass，见 01 §2 强类型配置解析）。

### 核心文件

| 文件 | 职责 |
|------|------|
| `hyper_models/components/datasets/llm/dataloader.py` | `build_dataloader()` 统一入口，内部 11 步流程（含可选 Transform/MultiSource/动态Batching 包装层） |
| `hyper_models/components/datasets/llm/chat_dataset.py` | ChatDataset 对话格式数据集（SFT），含 ShareGPT 格式转换 |
| `hyper_models/components/datasets/llm/formatting_utils.py` | Chat Template 格式化、loss mask 构建（`format_chat_template`） |
| `hyper_models/components/datasets/llm/packed_sequence.py` | THD packing |
| `hyper_models/components/datasets/llm/neat_packing.py` | NEAT packing (VLM) |
| `hyper_models/components/datasets/llm/megatron_dataset.py` | Megatron `.bin/.idx` 数据集封装 |
| `hyper_models/components/datasets/llm/megatron/sampler.py` | Megatron `MegatronPretrainingSampler` / `MegatronPretrainingRandomSampler` |
| `hyper_models/components/datasets/llm/eval/hellaswag.py` | HellaSwag 评估数据集 |
| `hyper_models/components/datasets/llm/eval/squad.py` | SQuAD 评估数据集 |
| `hyper_models/components/datasets/transforms/registry.py` | TransformRegistry 数据变换注册表 |
| `hyper_models/components/datasets/transforms/plaintext.py` | 纯文本变换 |
| `hyper_models/components/datasets/transforms/conversation.py` | 对话格式变换 |
| `hyper_models/components/datasets/transforms/dpo.py` | DPO 变换 |
| `hyper_models/components/datasets/transforms/vlm.py` | VLM 变换（qwen2_vl 等） |
| `hyper_models/components/datasets/multisource.py` | `WeightedMultiSourceDataset` 多源加权采样 + `_MapStyleIterableWrapper`（map-style 源适配） |
| `hyper_models/components/datasets/dynamic_batching.py` | 动态 batching（DynBszBuffer、TokenBasedBatchIterator） |
| `hyper_models/components/datasets/prefetch.py` | BackgroundPrefetcher 后台线程预取 |
| `hyper_models/components/datasets/reservoir_sampler.py` | 流式 shuffle 缓冲（ReservoirSampler） |
| `hyper_models/components/datasets/lazy_mapped_dataset.py` | 延迟映射 + LRU 缓存 |
| `hyper_models/components/datasets/vlm/loader.py` | VLM 数据加载器配置（VlmDataloaderConfig） |
| `hyper_models/components/datasets/vlm/datasets.py` | VLM 数据集工厂函数 |
| `hyper_models/components/datasets/vlm/neat_packing_vlm.py` | VLM NEAT packing |
| `hyper_models/components/datasets/vlm/media_utils.py` | VLM 媒体工具（图像/视频/音频加载与预处理） |
| `hyper_models/components/datasets/vlm/collate_fns.py` | VLM collate 函数（pad_collate_fn、neat_packed_vlm_collater） |
| `hyper_models/components/datasets/vlm/pp_media.py` | PP 模式下媒体 tensor 分片 |
| `hyper_models/components/datasets/vlm/fake_image.py` | 纯文本样本 fake image 注入（FSDP batch 结构一致性，§19 helper 签名） |
| `hyper_models/components/datasets/utils.py` | Collate 函数集合（default_collater、packed_sequence_thd_collater、neat_packed_collater、add_causal_masks_to_batch）+ helper（_get_model_name、compute_trust_remote_code_from_model、_should_precompute_pp_causal_masks） |
| `hyper_models/components/datasets/config.py` | 配置类型化解析层（DatasetConfig、PackingConfig 类型系统） |
| `hyper_models/components/datasets/llm/chat_templates.py` | Chat Template 注册表（CHAT_TEMPLATE_REGISTRY） |
| `hyper_models/components/datasets/llm/length_grouped_sampler.py` | LengthGroupedSampler（按长度分组采样） |
| `hyper_models/components/utils/model_utils.py` | 模型工具函数（`_supports_seq_lens`） |
| `hyper_models/components/models/common/packing.py` | Packing 配置模块（`configure_packing`、`get_attn_implementation`） |

### 涉及删除的旧代码

| 旧代码 | 替代方案 |
|--------|---------|
| `hyper_parallel/data/registry.py` — `DATASET_REGISTRY` 装饰器注册 | HF `datasets.load_dataset()` + `_target_` + TransformRegistry |
| `hyper_parallel/data/dummy.py` | 封装为 `_target_: hyper_models.components.datasets.llm.DeterministicTokenDataset` |
| `hyper_parallel/data/hf.py` | 合并到 `build_dataloader()`，直接使用 `datasets.load_dataset()` |
| `hyper_parallel/data/preset_pt.py` | 封装为 `torch.utils.data.TensorDataset` 或独立路径 |
| `hyper_parallel/trainer/llm_trainer.py::_build_data_transform()` | 移到 TransformRegistry 注册表 |
| `hyper_parallel/trainer/llm_trainer.py::_build_collate_fn()` | 移到 `hyper_models/components/datasets/utils.py` |

> **Megatron 数据源去留声明**：`hyper_parallel/data/megatron/` 下的
> `builder.py`（函数式 `build_megatron`）、`gpt_dataset.py`（`GPTDataset`）、
> `blendable_dataset.py`（`BlendableDataset`）、`indexed_dataset.py` **保留**，作为本文档新设计
> `MegatronPretraining`（§9）的底层实现——`MegatronPretraining.build()` 内部复用本地
> `GPTDataset`/`BlendableDataset`/`indexed_dataset` 完成二进制索引加载与样本切分，仅在外层
> 包装 `_target_` IoC 与 sampler 适配。即不删除本地 megatron 源码、也不 port AutoModel 全套，
> 而是让新 `MegatronPretraining` 复用本地实现。

---

### 命名空间与重命名声明

> **架构决策**：本文档所有代码路径统一使用 `hyper_models.components.datasets...`
> 命名空间（与真实 AutoModel 的 `nemo_automodel.components.datasets...` 对应）。
> 下文"与真实 `nemo_automodel/...` 对齐"的注释仅指对照真实源码核验签名，不代表
> 运行时 import `nemo_automodel`。
>
> **Tokenizer 重命名**：真实 AutoModel 的 `NeMoAutoTokenizer`（`nemo_automodel.
> _transformers.auto_tokenizer.NeMoAutoTokenizer`）在 hyper_parallel 中重命名为
> **`HyperAutoTokenizer`**（`hyper_models._transformers.auto_tokenizer.
> HyperAutoTokenizer`），二者等价。下文所有 `HyperAutoTokenizer.from_pretrained`
> 调用对应真实代码的 `NeMoAutoTokenizer.from_pretrained`。

---

## 2. 总入口调用时序：从 `recipe.setup()` 到 DataLoader 就绪

数据管道的全部构建工作在 `recipe.setup()` 中的 `build_dataloader()` 一次调用完成。以下是完整的调用树，数字序号表示执行顺序，缩进表示调用深度。

```
③ recipe.setup(cfg)                                                  # 03_training_loop.md（D1 时序编号，以 01 §4.1 为 canonical）
│
├─ ... (model, optimizer, loss, ... 等组件构建)
│
└─③.8 self.dataloader, self.tokenizer = build_dataloader(  # ③.8 = 01 §4.1 canonical 编号（以 01 为准）; ⑧ = 本文档内部步骤编号
        cfg.dataset, cfg.dataloader, cfg.model, cfg.packed_sequence,
        cfg.dynamic_batching, cfg.transform, cfg.multisource,        # （规划中：数据管道配置段，经 resolve_data_config 独立解析，不在当前 TrainerConfig 字段内）
        seed, local_batch_size, global_batch_size,
        max_steps, val_check_interval, dp_rank, dp_world_size,
        pp_enabled, cp_size, model)
    │
    ├─⑧.1 kwargs, tokenizer = _build_tokenizer(cfg_model, cfg_ds)    # §4: 4 路分发
    │
    ├─⑧.2 TransformRegistry 包装（可选）                              # §5: 新增
    │   ├─ transform_name = cfg_transform.get("name", None)
    │   └─ ds = LazyMappedDataset(ds, TransformRegistry.get(transform_name)(..., tokenizer))
    │
    ├─⑧.3 MultiSource 包装（可选）                                    # §6: 新增
    │   └─ ds = WeightedMultiSourceDataset(sources=..., weights=...)
    │
    ├─⑧.4 ds = dataset_target(**kwargs)                            # §7: 数据集实例化
    │   │   （dataset_target = import_target(cfg.dataset["_target_"]) 解析出的
    │   │    callable；解析入口为规划中的 resolve_data_config()，
    │   │    机制复用 01 §2.4 的 import_target() + coerce_value()）
    │   ├─ dataset_target == ChatDataset                             # §7.1: 对话格式
    │   │   → ChatDataset(path_or_dataset_id, tokenizer, ...)
    │   ├─ dataset_target == MegatronPretraining                     # §7.2: Megatron 路径
    │   │   → MegatronPretraining(paths=[...], ...).build()
    │   └─ dataset_target == datasets.load_dataset (或其他)          # §7.3: HF 路径
    │       → load_dataset(path="HuggingFaceFW/fineweb", ...)
    │
    ├─⑧.5 IterableDataset 分片                                        # §8
    │   ├─ ds.shard(dp_world_size, dp_rank)          if callable
    │   └─ split_dataset_by_node(ds.dataset, ...)     if HF streaming
    │
    ├─⑧.6 Packed Sequence（可选）                                      # §9
    │   ├─ packing_strategy == "neat" → neat_pack_dataset(...)
    │   └─ else → pack_dataset(...)
    │
    ├─⑧.7 Sampler                                                      # §10
    │   ├─ MegatronPretraining → create_megatron_sampler(...)
    │   ├─ map-style Dataset → StatefulDistributedSampler(...)
    │   └─ IterableDataset → 不设 sampler/batch_size
    │
    ├─⑧.8 Collate 函数                                                 # §11
    │   ├─ cfg_dl.collate_fn 有 _target_  → import_target 解析后构建期调用一次
    │   │     （嵌套 _target_ 段由 resolve_data_config 递归解析，规划中）
    │   ├─ cfg_dl.collate_fn 是 callable  → 直接用
    │   ├─ 否则 → default_collater(tokenizer)
    │   └─ PP 模式 → chained_collate_fn（base → add_causal_masks_to_batch）
    │
    ├─⑧.9 构建 DataLoader                                             # 最终一步
    │   → StatefulDataLoader(dataset=ds, sampler=sampler,
    │                        batch_size=1, collate_fn=<fn>, ...)
    │
    ├─⑧.10 动态 Batching 包装（可选）                                  # §12: 新增
    │   → TokenBasedBatchIterator(dl, dynamic_batching_config)
    │
    └─⑧.11 return dl, tokenizer
```

**与 01 文档的时序衔接**：

```
main()
├─① cfg = parse_training_args()   # 01 §2: 强类型配置解析 → TrainerConfig
│   # 注：TrainerConfig 当前只接受 9 个一级字段（model/optimizer/lr_scheduler/
│   # loss/training/accelerator/mixed_precision/gradient_checkpointing/debug），
│   # resolve_root() 拒绝未知一级字段；dataset/dataloader/packed_sequence 等
│   # 数据管道配置段须经 resolve_data_config() 独立解析（规划中），不走 resolve_root()
├─② recipe = FinetuneRecipe()     # 01 §4.1（规划中）
└─③ recipe.setup(cfg)            # 01 §4.1; cfg 为 TrainerConfig（不再需要 RecipeConfig 桥接）
    ├─③.4  model = ...           # 01 §4.1/§6
    ├─③.6  optimizer = ...       # 03_training_loop §9
    └─③.8  dataloader, tokenizer = build_dataloader(...)  ← 本文档入口（编号以 01 §4.1 为准）
```

---

## 3. build_dataloader() 主流程

> **调用位置**: 时序树 ⑧ — `recipe.setup()` 中唯一入口，一次返回 `(DataLoader, tokenizer)`

### 3.1 函数签名

```python
# hyper_models/components/datasets/llm/dataloader.py

def build_dataloader(
    cfg_ds,                # Dataset 配置（含 _target_）
    cfg_dl,                # DataLoader 配置
    cfg_model,             # Model 配置
    cfg_ps,                # PackedSequence 配置
    seed: int,
    local_batch_size: int,
    global_batch_size: int,
    cfg_db=None,           # DynamicBatching 配置（可选，新增）
    cfg_transform=None,    # Transform 配置（可选，新增）
    cfg_multisource=None,  # MultiSource 配置（可选，新增）
    *,                   # 以下均为 keyword-only（调用方 03 按关键字传参）
    max_steps: int | None = None,
    val_check_interval: int | None = None,
    dp_rank: int,
    dp_world_size: int,
    pp_enabled: bool,
    cp_size: int = 1,
    model: nn.Module | None = None,
) -> tuple[DataLoader, PreTrainedTokenizerBase]:
    """构建 DataLoader 的统一入口。

    支持的数据集类型（通过 cfg_ds._target_ 自动分发）：
    - ChatDataset → 对话格式 SFT 数据集
    - datasets.load_dataset() → HF hub 数据集
    - MegatronPretraining → Megatron .bin/.idx 格式
    - 自定义 Dataset 类 → 任意 `_target_` 可调用的 Dataset

    新增可选包装层：
    - cfg_transform: TransformRegistry 数据变换
    - cfg_multisource: WeightedMultiSourceDataset 多源加权采样
    - cfg_db: TokenBasedBatchIterator 动态 batching

    返回: (DataLoader, tokenizer)
    """
```

### 3.2 完整实现（11 步流程）

```python
# 模块级 import：
# import logging, random, inspect
# from transformers import AutoConfig
# from transformers.tokenization_utils_base import PreTrainedTokenizerBase
# from torch.utils.data import DataLoader, IterableDataset
# from hyper_models.components.datasets.llm.chat_dataset import ChatDataset
# from hyper_models.components.datasets.llm.megatron_dataset import MegatronPretraining
# from hyper_models.components.datasets.llm.megatron.sampler import create_megatron_sampler
# from hyper_models.components.datasets.llm.packed_sequence import pack_dataset
# from hyper_models.components.datasets.multisource import (
#     WeightedMultiSourceDataset, _MapStyleIterableWrapper,
# )
# from hyper_models.components.datasets.dynamic_batching import TokenBasedBatchIterator
# from hyper_models.components.datasets.transforms.registry import TransformRegistry
# from hyper_models.components.datasets.lazy_mapped_dataset import LazyMappedDataset
# from hyper_models.components.datasets.utils import (
#     default_collater, packed_sequence_thd_collater, neat_packed_collater,
#     add_causal_masks_to_batch,
# )
# from hyper_models.components.training.rng import ScopedRNG
# from hyper_models.components.distributed.utils import FirstRankPerNode
# from hyper_models.components.utils.model_utils import _supports_seq_lens
# 注：ConfigNode 已不存在，配置容器均为强类型 dataclass（见 01 §2）
# from hyper_models.config.resolver import import_target  # _target_ 解析（01 §2.4；
#     数据管道各配置段规划上由 resolve_data_config() 统一调用，见 §18）
# from hyper_models.components.datasets.utils import (
#     _get_model_name, compute_trust_remote_code_from_model,
#     _should_precompute_pp_causal_masks,
# )
# from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
# from torchdata.stateful_dataloader import StatefulDataLoader
#
# logger = logging.getLogger(__name__)

def build_dataloader(
    cfg_ds, cfg_dl, cfg_model, cfg_ps,
    seed, local_batch_size, global_batch_size,
    cfg_db=None, cfg_transform=None, cfg_multisource=None,
    *,  # 以下 keyword-only
    max_steps=None, val_check_interval=None,
    dp_rank, dp_world_size, pp_enabled, cp_size=1,
    model=None,
) -> tuple[DataLoader, PreTrainedTokenizerBase]:
    """构建 DataLoader。"""

    with ScopedRNG(seed=seed, ranked=True):
        # ── Step 1: 构建 Tokenizer ──
        kwargs, tokenizer = _build_tokenizer(cfg_model, cfg_ds)

        # dl_kwargs 在 Step 7（NEAT packing）就可能写入 collate_fn，
        # 必须在首次使用前初始化；Step 8 各分支一律 update，不整体重赋值
        dl_kwargs: dict = {}

        # ── Step 2: TransformRegistry 数据变换包装（可选） ──
        # 如果配置了 transform，在原始 dataset 上包装 LazyMappedDataset
        # 注意：此步在 dataset 实例化之前，用于声明式变换；
        # 与 dataset 内部的 __getitem__ 变换不冲突。
        transform_name = cfg_transform.get("name", None) if cfg_transform else None
        transform_kwargs = cfg_transform.to_dict() if cfg_transform else {}
        if transform_name:
            transform_fn = TransformRegistry.get(transform_name)
            # 将 tokenizer 和 max_seq_length 等参数绑定到 transform_fn
            # 使用默认参数绑定（而非闭包捕获）确保值在定义时冻结，避免后续
            # tokenizer/cfg_ds 变化导致不一致
            _seq_len = cfg_ds.get("seq_length", 8192)
            _extra_kwargs = {k: v for k, v in transform_kwargs.items() if k != "name"}
            bound_transform = lambda raw, tk=tokenizer, msl=_seq_len, ekw=_extra_kwargs: (
                transform_fn(raw, tokenizer=tk, max_seq_length=msl, **ekw)
            )
            # 使用 LazyMappedDataset 包装：延迟变换，不提前 tokenize
            # 先实例化 dataset，再包装
            # 注意：此处仅声明变换函数和参数；实际 LazyMappedDataset 包装在
            # Step 3（multisource 模式，按源包装）或 Step 5（单源模式）执行
            # （因为此时 ds 尚未实例化）

        # ── Step 3: MultiSource 包装（可选） ──
        # 如果配置了 multisource，则在 build_dataloader 外部构建
        # 多个子数据集，然后包装为 WeightedMultiSourceDataset
        # 注意：WeightedMultiSourceDataset 的构造需要前置的多个子数据集
        # 都构建好，因此需要特殊处理。
        # 与 automodel 的 DataloaderConfig 不同，此处沿用 VeOmni 的
        # WeightedMultiSourceDataset 模式，但通过 _target_ 配置。
        if cfg_multisource:
            # 多源模式：cfg_multisource 包含 sources 列表
            # 每个 source 有自己的 _target_ 和参数
            # 此处解析每个 source 并构建独立的 Dataset
            sources = []
            weights = []
            for src_cfg in cfg_multisource.get("sources", []):
                src_kwargs = dict(kwargs)
                # weight 是 multisource 的保留键，不属于子数据集构造参数。
                # _target_ 调用时会把配置中所有非保留键传给 target，
                # 因此实例化前必须剥离，避免 weight 泄漏进子数据集构造函数导致 TypeError
                weight = src_cfg.get("weight", 1.0)
                src_cfg_clean = {k: v for k, v in src_cfg.items()
                                 if k not in ("weight", "_target_")}
                # 嵌套 _target_ 段由 resolve_data_config（规划中）递归解析为
                # callable，机制复用 01 §2.4 的 import_target() + coerce_value()；
                # 此处以显式 import_target() 示意解析结果
                src_target = import_target(src_cfg["_target_"])
                if src_target is ChatDataset:
                    src = src_target(**src_cfg_clean, **{
                        "tokenizer": tokenizer,
                        "seq_length": src_cfg.get("seq_length", cfg_ds.get("seq_length", 8192)),
                    })
                elif src_target is MegatronPretraining:
                    # Megatron 源：与 Step 4 同口径注入训练参数，
                    # build() 后取底层 GPTDataset/BlendableDataset
                    src_kwargs.update({
                        "global_batch_size": global_batch_size,
                        "micro_batch_size": local_batch_size,
                        "trainer_max_steps": max_steps,
                        "trainer_val_check_interval": val_check_interval,
                    })
                    src = src_target(**src_cfg_clean, **src_kwargs)
                    src.build()
                    src = src.get_dataset(split=src_cfg.get("split", "train"))
                else:
                    with FirstRankPerNode():
                        src = src_target(**src_cfg_clean, **src_kwargs)

                # transform 配置存在时按源包装（替代单源模式 Step 5 的整体包装）；
                # LazyMappedDataset 为 map-style，Iterable 源暂不支持，显式报错
                if transform_name:
                    if isinstance(src, IterableDataset):
                        raise NotImplementedError(
                            "transform + multisource 暂不支持 IterableDataset 源"
                        )
                    src = LazyMappedDataset(
                        dataset=src, map_fn=bound_transform,
                        cache_size=cfg_transform.get("cache_size", 10000),
                    )

                # WeightedMultiSourceDataset 按 iter() 消费各源（§6.2），
                # map-style 源（ChatDataset/GPTDataset）需包装为 Iterable；
                # DP 分片由外层 WeightedMultiSourceDataset 统一负责
                # （upstream_sharded=False），包装器不做分片
                if not isinstance(src, IterableDataset):
                    src = _MapStyleIterableWrapper(src, seed=seed)
                sources.append(src)
                weights.append(weight)

            ds = WeightedMultiSourceDataset(
                datasets=sources,
                weights=weights,
                seed=cfg_multisource.get("seed", seed),
                level=cfg_multisource.get("level", "sample"),
                stopping_strategy=cfg_multisource.get("stopping_strategy", "first_exhausted"),
                # DP 信息显式传入（hyper_parallel 无 VeOmni 的
                # get_parallel_state() 全局态，见 §6.2 适配说明）
                dp_size=dp_world_size,
                dp_rank=dp_rank,
            )
        else:
            # ── Step 4: 实例化 Dataset ──
            # dataset 段的 _target_ 为字符串 dotted path，经 import_target() 解析
            # 为 callable 后按类型分发；解析入口为规划中的 resolve_data_config()
            # （机制复用 01 §2.4 import_target() + coerce_value()），
            # 此处以显式 import_target() 示意解析结果
            dataset_target = import_target(cfg_ds["_target_"])
            if dataset_target is ChatDataset:
                # ChatDataset 路径：需要 tokenizer
                # 从 cfg_ds 提取 path_or_dataset_id（核心字段），其余 kwargs 由
                # _build_tokenizer 产出的 kwargs 提供基础键（tokenizer）；
                # seq_length/padding/truncation/mask_* 等从 cfg_ds 读取
                path_or_dataset_id = cfg_ds.get("path_or_dataset_id")
                if not path_or_dataset_id:
                    raise ValueError(
                        "ChatDataset requires 'path_or_dataset_id' in dataset config"
                    )
                kwargs["tokenizer"] = tokenizer
                kwargs["seq_length"] = cfg_ds.get("seq_length", 8192)
                kwargs["padding"] = cfg_ds.get("padding", "do_not_pad")
                kwargs["truncation"] = cfg_ds.get("truncation", "do_not_truncate")
                kwargs["mask_history"] = cfg_ds.get("mask_history", False)
                kwargs["mask_reasoning_content"] = cfg_ds.get("mask_reasoning_content", False)
                ds = ChatDataset(path_or_dataset_id, **kwargs)
            elif dataset_target is MegatronPretraining:
                # Megatron 路径
                kwargs["global_batch_size"] = global_batch_size
                kwargs["micro_batch_size"] = local_batch_size
                kwargs["trainer_max_steps"] = max_steps
                kwargs["trainer_val_check_interval"] = val_check_interval
                ds = dataset_target(**kwargs)
                ds.build()
            else:
                # HF datasets 路径：每节点仅 rank 0 触发下载
                with FirstRankPerNode():
                    ds = dataset_target(**kwargs)

        # ── Step 5: 应用 Transform（单源模式；multisource 已在 Step 3 按源包装） ──
        if transform_name and not cfg_multisource:
            ds = LazyMappedDataset(
                dataset=ds,
                map_fn=bound_transform,
                cache_size=cfg_transform.get("cache_size", 10000),
            )

        # ── Step 6: IterableDataset 分片 ──
        if isinstance(ds, IterableDataset):
            if callable(getattr(ds, "shard", None)):
                ds = ds.shard(dp_world_size, dp_rank)
            elif hasattr(ds, "dataset"):
                from datasets.distributed import split_dataset_by_node
                ds.dataset = split_dataset_by_node(
                    ds.dataset, world_size=dp_world_size, rank=dp_rank
                )

        # ── Step 7: Packed Sequence（可选） ──
        packed_sequence_size = getattr(cfg_ps, "packed_sequence_size", 0)
        packing_strategy = getattr(cfg_ps, "packing_strategy", "thd")
        prepacked_sequence = bool(getattr(cfg_ps, "prepacked", False))

        supports_seq_lens = model is not None and _supports_seq_lens(model)
        if packed_sequence_size > 0 and packing_strategy == "thd" and not supports_seq_lens:
            logger.warning("Packed sequence is not supported without seq_lens; disabling")
            packed_sequence_size = 0

        if packed_sequence_size > 0 and prepacked_sequence:
            logger.info("Using prepacked sequence dataset; skipping recipe-side packing")
        elif packed_sequence_size > 0:
            if hasattr(ds, "shuffle"):
                ds = ds.shuffle(seed)
            if packing_strategy == "neat":
                from hyper_models.components.datasets.llm.neat_packing import neat_pack_dataset
                from hyper_models.components.datasets.utils import neat_packed_collater
                from hyper_models.components.models.common.packing import (
                    configure_packing, get_attn_implementation,
                )
                ds = neat_pack_dataset(
                    ds, split=cfg_ds.get("split", "train"),
                    pack_size=packed_sequence_size,
                    max_packs=getattr(cfg_ps, "max_packs", None),
                    padding_idx=getattr(tokenizer, "pad_token_id", 0),
                    drop_long_samples=getattr(cfg_ps, "drop_long_samples", True),
                )
                _attn_impl = get_attn_implementation(cfg_model)
                configure_packing(attn_implementation=_attn_impl)
                # 不修改全局 cfg_dl：直接将 collate_fn 写入 dl_kwargs，
                # Step 9（Collate）检测到 dl_kwargs 已有 collate_fn 时会跳过默认赋值
                dl_kwargs["collate_fn"] = lambda batch, _ai=_attn_impl: neat_packed_collater(
                    batch, attn_implementation=_ai
                )
            else:
                ds = pack_dataset(
                    ds, split=cfg_ds.get("split", "train"),
                    packed_sequence_size=packed_sequence_size,
                    max_packs=getattr(cfg_ps, "max_packs", None),
                    padding_idx=getattr(tokenizer, "pad_token_id", 0),
                    cp_size=cp_size,
                )

        # ── Step 8: Sampler ──
        if isinstance(ds, MegatronPretraining):
            split_to_get = cfg_ds.get("splits_to_build", None)
            if split_to_get is None:
                split_to_get = "train"
            elif isinstance(split_to_get, list):
                split_to_get = split_to_get[0]
            ds = ds.get_dataset(split=split_to_get)
            dataloader_type = cfg_dl.get("dataloader_type", "single")
            batch_sampler = create_megatron_sampler(
                dataset_len=len(ds),
                micro_batch_size=local_batch_size,
                global_batch_size=global_batch_size,
                dataloader_type=dataloader_type,
                rank=dp_rank, world_size=dp_world_size,
            )
            dl_kwargs["batch_sampler"] = batch_sampler
        elif not isinstance(ds, IterableDataset):
            shuffle = cfg_dl.get("shuffle", True)
            group_by_length = cfg_dl.get("group_by_length", False)
            drop_last = cfg_dl.get("drop_last", True)

            if group_by_length:
                from hyper_models.components.datasets.llm.length_grouped_sampler import (
                    LengthGroupedSampler as LLMLengthGroupedSampler,
                )
                sampler = LLMLengthGroupedSampler(
                    dataset=ds, batch_size=local_batch_size,
                    seed=seed, num_replicas=dp_world_size, rank=dp_rank,
                )
            else:
                sampler = StatefulDistributedSampler(
                    ds, seed=seed, drop_last=drop_last,
                    num_replicas=dp_world_size, rank=dp_rank,
                    shuffle=shuffle,
                )
            dl_kwargs.update({
                "sampler": sampler,
                "batch_size": local_batch_size,
                "drop_last": drop_last or pp_enabled,
            })
        else:
            shuffle = cfg_dl.get("shuffle", False)
            shuffle_buffer_size = cfg_dl.get("shuffle_buffer_size", 10000)
            if shuffle and hasattr(ds, "shuffle"):
                try:
                    ds = ds.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
                except Exception as e:
                    logger.warning(f"IterableDataset shuffle skipped: {e}")
            # IterableDataset 无需 sampler/batch_size，dl_kwargs 保持
            # Step 7 可能已写入的 collate_fn（NEAT）即可

        dl_kwargs["dataset"] = ds

        # ── Step 9: Collate ──
        seq_divisor = 2 * cp_size

        # 如果 Step 7（Packing）已经设置了 collate_fn（NEAT 路径），
        # 跳过默认 collate 赋值；否则按配置解析 collate_fn
        if "collate_fn" not in dl_kwargs:
            if hasattr(cfg_dl, "collate_fn"):
                if getattr(cfg_dl.collate_fn, "_target_", None) is not None:
                    # 嵌套 _target_ 段由 resolve_data_config（规划中）递归解析，
                    # 机制复用 01 §2.4 import_target() + coerce_value()；
                    # _target_ 声明的 collator 在构建期实例化一次，得到实现了
                    # __call__(batch) 的可调用对象；不能在每个 batch 到来时
                    # 才实例化（那会把 collator 实例误当 collate 结果）
                    collate_cfg = cfg_dl.collate_fn.to_dict()
                    collate_target = import_target(collate_cfg.pop("_target_"))
                    dl_kwargs["collate_fn"] = collate_target(**collate_cfg)
                else:
                    dl_kwargs["collate_fn"] = cfg_dl.collate_fn
                assert callable(dl_kwargs["collate_fn"]), "collate_fn must be callable"
            else:
                if tokenizer is None:
                    logger.warning("No tokenizer configured; default_collater falls back to pad_token_id=0")
                dl_kwargs["collate_fn"] = default_collater(tokenizer, seq_divisor=seq_divisor)

        # transform 路径下，LazyMappedDataset.__getitem__ 可能返回
        # list[dict]（1→N 分块变换，§5.1），collate 入口需先展平为
        # 扁平 list[dict] 再交给底层 collate_fn
        if transform_name:
            base_collate_fn = dl_kwargs["collate_fn"]
            def flatten_collate(batch, base_fn=base_collate_fn):
                flat = []
                for item in batch:
                    if isinstance(item, (list, tuple)):
                        flat.extend(item)
                    else:
                        flat.append(item)
                return base_fn(flat)
            dl_kwargs["collate_fn"] = flatten_collate

        if pp_enabled:
            from hyper_models.components.datasets.utils import add_causal_masks_to_batch
            try:
                hf_model_config = AutoConfig.from_pretrained(
                    _get_model_name(cfg_model),
                    trust_remote_code=compute_trust_remote_code_from_model(cfg_model),
                )
            except Exception:
                logger.warning("Failed to load model config for causal mask precomputation; skipping")
                hf_model_config = None

            if hf_model_config is not None and _should_precompute_pp_causal_masks(hf_model_config):
                if "collate_fn" in dl_kwargs:
                    base_collate_fn = dl_kwargs["collate_fn"]
                    def chained_collate_fn(batch, base_fn=base_collate_fn, config=hf_model_config):
                        batch = base_fn(batch)
                        return add_causal_masks_to_batch(batch, model_config=config)
                    dl_kwargs["collate_fn"] = chained_collate_fn

        # ── Step 10: 构建 DataLoader ──
        # 从 cfg_dl 提取 DataLoader 构造合法的 kwargs（过滤掉已在前面步骤
        # 单独处理的键），避免将 sampler 专用键（shuffle/drop_last 等）传入
        # StatefulDataLoader 构造函数。
        _DL_HANDLED_KEYS = {
            "shuffle", "group_by_length", "drop_last", "dataloader_type",
            "shuffle_buffer_size", "collate_fn", "_target_",
        }
        if hasattr(cfg_dl, "to_dict"):
            dl_base_kwargs = {
                k: v for k, v in cfg_dl.to_dict().items()
                if k not in _DL_HANDLED_KEYS and k not in dl_kwargs
            }
        else:
            dl_base_kwargs = {}
        dl_base_kwargs.update({
            "num_workers": cfg_dl.get("num_workers", 1),
            "pin_memory": cfg_dl.get("pin_memory", True),
        })
        if (
            "drop_last" not in dl_kwargs
            and "batch_sampler" not in dl_kwargs
            and cfg_dl.get("drop_last", True)
        ):
            dl_base_kwargs["drop_last"] = True
        # dl_kwargs 包含 sampler/batch_sampler/collate_fn/dataset，
        # dl_base_kwargs 包含 num_workers/pin_memory 等通用参数
        dl = StatefulDataLoader(**dl_kwargs, **dl_base_kwargs)

        # ── Step 11: 动态 Batching 包装（可选） ──
        if cfg_db and cfg_db.get("enabled", False):
            from hyper_models.components.datasets.dynamic_batching import (
                TokenBasedBatchIterator, DynamicBatchingConfig,
            )
            db_config = DynamicBatchingConfig(
                enabled=cfg_db.get("enabled", True),
                n_token_per_iter=cfg_db.get("n_token_per_iter", local_batch_size * cfg_ds.get("seq_length", 8192)),
                buffer_size=cfg_db.get("buffer_size", 10000),
                bsz_warmup_steps=cfg_db.get("bsz_warmup_steps", 0),
                bsz_warmup_init_mbtoken=cfg_db.get("bsz_warmup_init_mbtoken", 0),
                physical_token_cap=cfg_db.get("physical_token_cap", 0),
                count_mode=cfg_db.get("count_mode", "total"),
            )
            dl = TokenBasedBatchIterator(dl, db_config)

        return dl, tokenizer
```

### 3.3 Validation DataLoader

```python
# hyper_models/components/datasets/llm/dataloader.py

def build_validation_dataloader(
    cfg_ds,
    cfg_dl,
    cfg_model,
    cfg_ps,
    seed: int,
    local_batch_size: int,
    global_batch_size: int,
    dp_rank: int,
    dp_world_size: int,
    pp_enabled: bool,
    cp_size: int = 1,
    model: nn.Module | None = None,
    drop_last: bool = False,
    shuffle: bool = False,
    no_packing: bool = True,
) -> dict[str, DataLoader]:
    """构建验证用 DataLoader。

    与 build_dataloader 的差异：
    - drop_last=False、shuffle=False、no_packing=True（默认）
    - 不创建 sampler 的断点续训状态（验证集无 resume 需求）
    - 不传 max_steps/val_check_interval（不影响采样调度）
    - 无动态 batching、无 multisource

    通过覆盖 cfg_ps.packed_sequence_size=0 关闭 packing，再委托 build_dataloader。

    Returns:
        {"validation": DataLoader}
    """
    # 注：配置容器支持 replace() 方法（01 §3 Configurable.Config.replace()），
    # dataclass 配置直接 replace() 创建覆盖式拷贝；plain dict 走浅拷贝合并
    def _override_cfg(cfg, **overrides):
        if hasattr(cfg, "replace"):
            # dataclass 配置（Configurable.Config）：不可变覆盖式拷贝
            return cfg.replace(**overrides)
        # plain dict：浅拷贝后合并 overrides
        return {**cfg, **overrides}

    cfg_ps_val = _override_cfg(cfg_ps, packed_sequence_size=0) if no_packing else cfg_ps
    cfg_dl_val = cfg_dl
    if cfg_dl.get("drop_last", True) != drop_last or cfg_dl.get("shuffle", True) != shuffle:
        cfg_dl_val = _override_cfg(cfg_dl, drop_last=drop_last, shuffle=shuffle)

    dl, _ = build_dataloader(
        cfg_ds, cfg_dl_val, cfg_model, cfg_ps_val,
        seed=seed,
        local_batch_size=local_batch_size,
        global_batch_size=global_batch_size,
        cfg_db=None, cfg_transform=None, cfg_multisource=None,
        max_steps=None, val_check_interval=None,
        dp_rank=dp_rank, dp_world_size=dp_world_size,
        pp_enabled=pp_enabled, cp_size=cp_size, model=model,
    )
    return {"validation": dl}
```

---

## 4. Tokenizer 构建

> **调用位置**: 时序树 ⑧.1 — `build_dataloader()` Step 1

### 4.1 设计理念

Tokenizer 的类型和来源通过 YAML `_target_` 声明。4 条构建路径：

1. **无 tokenizer key** → 从 model 推断 → `HyperAutoTokenizer.from_pretrained(model_name)`
2. **tokenizer 为 null** → 跳过
3. **有 tokenizer 但无 `_target_`** → `HyperAutoTokenizer.from_pretrained(**tokenizer_dict)`
4. **有 `_target_`** → `import_target(cfg_ds.tokenizer["_target_"])(..., trust_remote_code=...)`（嵌套 `_target_` 段由 resolve_data_config 递归解析，规划中）

```yaml
dataset:
  tokenizer:
    _target_: transformers.AutoTokenizer.from_pretrained
    pretrained_model_name_or_path: Qwen/Qwen3.5-4B
    trust_remote_code: true
```

### 4.2 实现：4 路分发

```python
def _build_tokenizer(cfg_model, cfg_ds) -> tuple[dict, PreTrainedTokenizerBase]:
    """从配置构建 tokenizer（强类型配置，`_target_` 声明 tokenizer 类型）。

    与 AutoModel train_ft.py::_build_tokenizer 完全对齐的 4 路分发。
    """
    from hyper_models._transformers.auto_tokenizer import HyperAutoTokenizer

    trust_remote_code = compute_trust_remote_code_from_model(cfg_model)

    # ── 路径 1: 无 tokenizer key → 从 model 推断 ──
    if "tokenizer" not in cfg_ds and _get_model_name(cfg_model) is not None:
        logging.info("Using model config to instantiate tokenizer")
        tokenizer = HyperAutoTokenizer.from_pretrained(
            _get_model_name(cfg_model), trust_remote_code=trust_remote_code
        )
    # ── 路径 2: tokenizer 显式为 null → 跳过 ──
    elif cfg_ds.get("tokenizer", None) is None:
        tokenizer = None
    # ── 路径 3: 有 tokenizer 但无 _target_ → from_pretrained(**dict) ──
    elif getattr(cfg_ds.tokenizer, "_target_", None) is None:
        tokenizer_dict = cfg_ds.tokenizer.to_dict()
        trust_remote_code = tokenizer_dict.pop("trust_remote_code", trust_remote_code)
        tokenizer = HyperAutoTokenizer.from_pretrained(
            **tokenizer_dict, trust_remote_code=trust_remote_code
        )
    # ── 路径 4: 有 _target_ → import_target 解析后显式调用 ──
    else:
        # 嵌套 _target_ 段由 resolve_data_config（规划中）递归解析，
        # 机制复用 01 §2.4 的 import_target() + coerce_value()
        tokenizer_cfg = cfg_ds.tokenizer.to_dict()
        trust_remote_code = tokenizer_cfg.pop("trust_remote_code", trust_remote_code)
        tokenizer_target = import_target(tokenizer_cfg.pop("_target_"))
        tokenizer = tokenizer_target(**tokenizer_cfg, trust_remote_code=trust_remote_code)

    # 设置 pad_token
    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset 构建时需要的 kwargs（根据 _target_ 签名决定是否注入 tokenizer）
    # dataset 段 _target_ 为字符串 dotted path，先经 import_target() 解析为
    # callable 再检查签名（解析入口 resolve_data_config 规划中）
    kwargs = {}
    ds_target = import_target(cfg_ds["_target_"]) if cfg_ds.get("_target_") else None
    if tokenizer is not None and ds_target is not None:
        try:
            sig = inspect.signature(ds_target)
            if "tokenizer" in sig.parameters:
                kwargs["tokenizer"] = tokenizer
        except (ValueError, TypeError):
            pass
    return kwargs, tokenizer
```

---

## 5. TransformRegistry 数据变换注册表

> **调用位置**: 时序树 ⑧.2 — `build_dataloader()` Step 2（可选）
> **参考**: VeOmni `veomni/data/data_transform.py` — `DATA_TRANSFORM_REGISTRY`

### 5.1 设计理念

TransformRegistry 将数据预处理（tokenize、chat template 应用、图像编码等）与 Dataset 解耦。变换函数接收原始样本，返回 `list[dict]`（每个 dict 是一个 tokenized 训练样本，支持从一个原始样本中分块处理长序列）。

**1→N 分块的下游契约**：`LazyMappedDataset.__getitem__` 是 1:1 映射，
1→N 变换时返回的是 `list[dict]` 而非单个 dict。`build_dataloader` Step 9
在配置了 transform 时自动为 collate_fn 包一层 flatten（把 batch 内的
list 项展平为扁平 `list[dict]` 再交给底层 collate），因此 collate 层
无需感知分块语义；但这也意味着一个 DataLoader batch 的实际样本数可能
大于 `batch_size`（token 数相应放大），配置 `local_batch_size` 时需留意。

与 VeOmni 的区别：
- Hyper-Parallel 的 transform 不直接包装 Dataset，而是通过 `LazyMappedDataset` 延迟应用
- VeOmni 的 transform 在 `build_dataset` 阶段直接传入，Hyper-Parallel 在 `build_dataloader` Step 2 配置、Step 5 应用（multisource 模式下在 Step 3 按源包装）

### 5.2 注册表实现

```python
# hyper_models/components/datasets/transforms/registry.py

class TransformRegistry:
    """数据变换注册表。

    与 VeOmni DATA_TRANSFORM_REGISTRY 对齐的注册表模式。
    变换函数签名: (raw_sample: dict, tokenizer, max_seq_length: int, **kwargs) -> list[dict]
    """

    _transforms: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        """注册装饰器。"""
        def decorator(func):
            cls._transforms[name] = func
            return func
        return decorator

    @classmethod
    def get(cls, name: str) -> Callable:
        """获取已注册的变换函数。"""
        if name not in cls._transforms:
            raise KeyError(
                f"Transform '{name}' not registered. "
                f"Available: {list(cls._transforms.keys())}"
            )
        return cls._transforms[name]

    @classmethod
    def list_available(cls) -> list[str]:
        return list(cls._transforms.keys())
```

### 5.3 内置变换

#### 5.3.1 纯文本变换

```python
# hyper_models/components/datasets/transforms/plaintext.py

@TransformRegistry.register("plaintext")
def transform_plaintext(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    text_keys: str | list[str] = "text",
    **kwargs,
) -> list[dict]:
    """纯文本变换：分块 + tokenize。

    与 VeOmni process_plaintext_example 对齐。
    从 example[text_keys] 读取文本，tokenize 后按 max_seq_length 分块。
    返回 list[dict]，每个 dict 含 input_ids/attention_mask/labels（1-D Tensor）。

    Args:
        example: 原始样本
        tokenizer: 用于 tokenize 的 HF tokenizer
        max_seq_length: 最大序列长度
        text_keys: 文本字段名（支持列表，取第一个存在的 key）

    Returns:
        [{input_ids: Tensor[N], attention_mask: Tensor[N], labels: Tensor[N]}, ...]
    """
    from hyper_models.components.datasets.utils import split_into_chunks

    # 获取文本
    if isinstance(text_keys, str):
        text_keys = [text_keys]
    text = None
    for key in text_keys:
        if key in example and example[key]:
            text = example[key]
            break
    if text is None:
        text = str(example)

    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]

    # 分块
    chunks = split_into_chunks(tokens, max_seq_length)

    results = []
    for chunk in chunks:
        input_ids = torch.tensor(chunk)
        labels = input_ids.clone()  # NTP: labels == input_ids
        results.append({
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
        })
    return results
```

#### 5.3.2 对话格式变换

```python
# hyper_models/components/datasets/transforms/conversation.py

@TransformRegistry.register("conversation")
def transform_conversation(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    text_keys: str | list[str] = "messages",
    mask_history: bool = True,
    mask_reasoning_content: bool = False,
    **kwargs,
) -> list[dict]:
    """对话格式变换：应用 chat template + loss mask。

    参考 VeOmni process_conversation_example 和 Automodel format_chat_template。
    从 example[text_keys] 读取 messages，应用 tokenizer.apply_chat_template()
    构建 input_ids 和 labels（只对 assistant 回复计算 loss）。

    Args:
        example: 原始样本，含 messages 字段
        tokenizer: 需有 chat_template
        max_seq_length: 最大序列长度
        text_keys: 消息字段名
        mask_history: 是否只对 assistant 回复计算 loss
        mask_reasoning_content: 是否屏蔽 reasoning 内容 loss

    Returns:
        [{input_ids: Tensor[N], attention_mask: Tensor[N], labels: Tensor[N], ...}]
    """
    from hyper_models.components.datasets.llm.formatting_utils import format_chat_template

    messages = example.get(text_keys[0] if isinstance(text_keys, list) else text_keys, example.get("messages", []))
    if not messages:
        # 兼容 ShareGPT 格式
        messages = example.get("conversations", [])

    # 调用 format_chat_template 构建 input_ids 和 labels
    result = format_chat_template(
        tokenizer=tokenizer,
        formatted_text=messages,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or 0,
        seq_length=max_seq_length,
        padding="max_length" if kwargs.get("pad_to_max_length", False) else "do_not_pad",
        truncation="truncation" if kwargs.get("do_truncate", True) else "do_not_truncate",
        answer_only_loss_mask=mask_history,
        mask_reasoning_content=mask_reasoning_content,
    )

    # 转换为 Tensor
    return [{
        "input_ids": torch.tensor(result["input_ids"]),
        "labels": torch.tensor(result.get("labels", result.get("loss_mask", result["input_ids"]))),
        "attention_mask": torch.tensor(result["attention_mask"]),
    }]
```

#### 5.3.3 DPO 变换

```python
# hyper_models/components/datasets/transforms/dpo.py

@TransformRegistry.register("dpo")
def transform_dpo(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    **kwargs,
) -> list[dict]:
    """DPO 变换：chosen/rejected 拼接成单个序列。

    与 VeOmni process_dpo_example 对齐。
    chosen 和 rejected 的 position_ids 在边界处重置，
    使 flash-attention 将它们视为两个独立序列。

    Returns:
        [{input_ids: Tensor[N], attention_mask: Tensor[N], labels: Tensor[N], position_ids: Tensor[N]}]
    """
    IGNORE_INDEX = -100

    # 构建 chosen 和 rejected 的 token ids
    prompt_ids = []
    if "chosen" in example and "rejected" in example:
        # 对话格式
        chat_template = kwargs.get("chat_template", None)
        if chat_template:
            from hyper_models.components.datasets.llm.formatting_utils import format_chat_template
            _ft_kwargs = dict(
                tokenizer=tokenizer,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or 0,
                seq_length=max_seq_length,
                padding="do_not_pad",
                truncation="do_not_truncate",
                answer_only_loss_mask=False,
            )
            chosen_result = format_chat_template(formatted_text=example["chosen"], **_ft_kwargs)
            rejected_result = format_chat_template(formatted_text=example["rejected"], **_ft_kwargs)
            chosen_ids = chosen_result["input_ids"]
            rejected_ids = rejected_result["input_ids"]
            # chat_template 路径下 prompt 长度由模板应用前的 prompt 文本
            # tokenize 得到（若有 prompt 字段）
            if "prompt" in example:
                prompt_text = example["prompt"]
                if isinstance(prompt_text, str):
                    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        else:
            # 纯文本格式
            prompt = example.get("prompt", "")
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            chosen_ids = prompt_ids + tokenizer.encode(example["chosen"], add_special_tokens=False) + [tokenizer.eos_token_id]
            rejected_ids = prompt_ids + tokenizer.encode(example["rejected"], add_special_tokens=False) + [tokenizer.eos_token_id]
    else:
        return []

    # 拼接 chosen 和 rejected
    c_len, r_len = len(chosen_ids), len(rejected_ids)
    input_ids = torch.tensor(chosen_ids + rejected_ids)

    # labels: chosen 部分保留，rejected 部分保留，prompt 部分设为 -100
    labels = input_ids.clone()
    # prompt 部分设为 -100（假设 prompt 在 chosen_ids 的开头）
    if "prompt" in example:
        prompt_len = len(prompt_ids)
        labels[:prompt_len] = IGNORE_INDEX
        labels[c_len:c_len + prompt_len] = IGNORE_INDEX

    # position_ids: 在边界处重置
    position_ids = torch.cat([
        torch.arange(c_len),
        torch.arange(r_len),
    ])

    return [{
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": torch.ones_like(input_ids),
        "position_ids": position_ids,
    }]
```

#### 5.3.4 VLM 变换

```python
# hyper_models/components/datasets/transforms/vlm.py

@TransformRegistry.register("qwen2_vl")
def transform_qwen2_vl(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    processor=None,
    **kwargs,
) -> list[dict]:
    """Qwen2-VL 变换：图像 + 文本拼接。

    与 VeOmni process_sample_qwen_vl 对齐。
    处理图像编码、image_grid_thw、position_id_func 等。
    需要 processor（含 image_processor + tokenizer + chat_template）。

    Returns:
        [{input_ids, attention_mask, labels, pixel_values, image_grid_thw, position_ids, ...}]
    """
    from hyper_models.components.datasets.vlm.media_utils import (
        _extract_media_from_conversations, smart_resize,
    )

    if processor is None:
        raise ValueError("VLM transform requires processor")

    # 提取对话和媒体
    conversation = example.get("messages", example.get("conversation", []))
    images, videos = _extract_media_from_conversations(conversation)

    # 应用 chat_template 获取文本
    text = processor.apply_chat_template(conversation, tokenize=False)

    # 通过 processor 编码（含图像预处理）
    result = processor(
        text=[text],
        images=images if images else None,
        videos=videos if videos else None,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
    )

    # 构建 labels
    labels = result["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    result["labels"] = labels

    # 移除 batch 维度（dataset 级别为单样本）
    result = {k: v.squeeze(0) if v.dim() > 1 and v.shape[0] == 1 else v
              for k, v in result.items()}

    return [result]
```

---

## 6. WeightedMultiSourceDataset 多源加权采样

> **调用位置**: 时序树 ⑧.3 — `build_dataloader()` Step 3（可选）
> **参考**: VeOmni `veomni/data/dataset.py` — `WeightedMultiSourceDataset`

### 6.1 设计理念

多源加权采样是大规模训练的核心能力。支持：
- 多源按权重采样（const / changing 调度）
- token 级别重加权（长样本获得更高概率）
- 3 种耗尽策略（`first_exhausted` / `all_exhausted` / `never_exhausted`）
- 完整 checkpoint 恢复（每源状态 + 随机状态）

### 6.2 实现

```python
# hyper_models/components/datasets/multisource.py

from typing import Sequence, Optional, Literal, Any, Callable
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import numpy as np
import random


class WeightedMultiSourceDataset(IterableDataset):
    """多源加权采样数据集。

    与 VeOmni WeightedMultiSourceDataset 对齐。
    每个源独立维护一个 IterableDataset，按权重采样源。

    三种耗尽策略：
    - first_exhausted: 任一源耗尽即停止
    - all_exhausted: 所有源耗尽才停止
    - never_exhausted: 耗尽后循环（从头开始）

    适配说明：VeOmni 通过全局 `get_parallel_state()` 获取 dp_rank/dp_size；
    hyper_parallel 无此全局态，改为构造参数显式传入（由 build_dataloader
    Step 3 注入 DP 维 rank/size）。
    """

    def __init__(
        self,
        datasets: Sequence[IterableDataset],
        weights: Sequence[float],
        seed: int = 42,
        level: Literal["sample", "token"] = "sample",
        sample_token_len_fn: Optional[Callable[[Any], float]] = None,
        source_names: Optional[Sequence[str]] = None,
        source_ids: Optional[Sequence[str]] = None,
        upstream_sharded: bool = False,
        stopping_strategy: Literal["first_exhausted", "all_exhausted", "never_exhausted"] = "first_exhausted",
        output_index_for_resume: bool = False,
        dp_size: int = 1,
        dp_rank: int = 0,
    ):
        self._datasets = list(datasets)
        self._weights = np.array(weights, dtype=np.float64)
        self._weights /= self._weights.sum()
        self._seed = seed
        self._level = level
        self._sample_token_len_fn = sample_token_len_fn or (lambda x: 1.0)
        self._source_names = source_names or [f"source_{i}" for i in range(len(datasets))]
        self._source_ids = source_ids or self._source_names
        self._upstream_sharded = upstream_sharded
        self._stopping_strategy = stopping_strategy
        self._output_index_for_resume = output_index_for_resume
        self._ds_num = len(datasets)
        # DP 分片参数（upstream_sharded=False 时在 __iter__ 中按样本号取模分片）
        self._dp_size = dp_size
        self._dp_rank = dp_rank

        # 运行时状态（每个 worker 独立）
        self._random_state: Optional[np.random.RandomState] = None
        self._iters: list = []
        self._epoch: int = 0
        self._exhausted: list[bool] = [False] * self._ds_num
        self._avg_len_sum: list[float] = [0.0] * self._ds_num
        self._avg_len_count: list[int] = [0] * self._ds_num
        self._global_sample_idx: int = 0
        self._just_resumed: bool = False

    def __iter__(self):
        worker_id = get_worker_info().id if get_worker_info() else 0

        if not self._just_resumed:
            from numpy.random import SeedSequence, RandomState
            seed_seq = SeedSequence([self._seed, self._epoch, worker_id])
            self._random_state = RandomState(seed_seq)
            self._exhausted = [False] * self._ds_num
            self._avg_len_sum = [0.0] * self._ds_num
            self._avg_len_count = [0] * self._ds_num
            self._global_sample_idx = 0
        else:
            self._just_resumed = False

        self._iters = [iter(ds) for ds in self._datasets]

        while True:
            ds_idx = self._random_state.choice(self._ds_num, p=self._runtime_weights())
            try:
                sample = self._next_sample(ds_idx)
            except StopIteration:
                break

            if sample is None:
                continue

            token_len = self._sample_token_len_fn(sample)
            if token_len <= 0:
                continue

            if self._level == "token":
                self._avg_len_sum[ds_idx] += token_len
                self._avg_len_count[ds_idx] += 1

            self._global_sample_idx += 1

            # DP sharding（如果上游未分片）
            if not self._upstream_sharded:
                if self._global_sample_idx % self._dp_size != self._dp_rank:
                    continue

            if self._output_index_for_resume:
                yield sample, (self._source_ids[ds_idx], self._global_sample_idx - 1)
            else:
                yield sample

    def _runtime_weights(self) -> np.ndarray:
        """计算运行时权重。

        level == "sample": 使用原始权重
        level == "token": 按平均长度重新加权（长样本获得更高概率）
        """
        if self._level == "sample":
            return self._weights

        # token 级别重加权
        weights = self._weights.copy()
        for i in range(self._ds_num):
            if self._avg_len_count[i] > 0:
                avg_len = self._avg_len_sum[i] / self._avg_len_count[i]
                weights[i] /= avg_len
        total = weights.sum()
        if total > 0:
            weights /= total
        return weights

    def _next_sample(self, ds_idx: int) -> Any:
        """从指定源取下一个样本。"""
        try:
            return next(self._iters[ds_idx])
        except StopIteration:
            if self._stopping_strategy == "first_exhausted":
                raise  # 任一源耗尽即停止
            elif self._stopping_strategy == "all_exhausted":
                self._exhausted[ds_idx] = True
                if all(self._exhausted):
                    raise
                return None  # 跳过，继续从其他源取
            else:  # never_exhausted
                self._exhausted[ds_idx] = True
                if all(self._exhausted):
                    # 所有源耗尽，全部重置从头开始
                    self._exhausted = [False] * self._ds_num
                    self._iters[ds_idx] = iter(self._datasets[ds_idx])
                else:
                    return None
                return next(self._iters[ds_idx])

    def state_dict(self) -> dict:
        """返回 checkpoint 状态。"""
        return {
            "version": 0,
            "runtime": {
                "random_state": self._random_state.get_state() if self._random_state else None,
                "avg_len_sum": self._avg_len_sum,
                "avg_len_count": self._avg_len_count,
                "exhausted": self._exhausted,
                "global_sample_idx": self._global_sample_idx,
                "dataset_states": [
                    ds.state_dict() if hasattr(ds, "state_dict") else ds.__getstate__()
                    for ds in self._datasets
                ],
            },
        }

    def load_state_dict(self, state: dict):
        """从 checkpoint 恢复。"""
        runtime = state["runtime"]
        if runtime["random_state"] is not None:
            self._random_state = np.random.RandomState()
            self._random_state.set_state(runtime["random_state"])
        self._avg_len_sum = runtime["avg_len_sum"]
        self._avg_len_count = runtime["avg_len_count"]
        self._exhausted = runtime["exhausted"]
        self._global_sample_idx = runtime["global_sample_idx"]
        for ds, ds_state in zip(self._datasets, runtime["dataset_states"]):
            if hasattr(ds, "load_state_dict"):
                ds.load_state_dict(ds_state)
            else:
                ds.__setstate__(ds_state)
        self._just_resumed = True


class _MapStyleIterableWrapper(IterableDataset):
    """将 map-style Dataset 包装为 IterableDataset（按 epoch 循环 + 每 epoch 重洗）。

    WeightedMultiSourceDataset.__iter__ 通过 `iter(ds)` / `next()` 消费各源，
    map-style 源（ChatDataset、GPTDataset 等）没有 __iter__，必须经本包装器
    适配后才能作为 multisource 的源（build_dataloader Step 3 自动包装）。

    DP 分片不在此处进行——由外层 WeightedMultiSourceDataset 按全局样本号
    取模分片（upstream_sharded=False），保证多源混合后的 DP 口径一致。

    耗尽语义：一个 epoch 迭代完即抛 StopIteration，由外层的
    stopping_strategy 决定停止还是进入下一 epoch（never_exhausted 时
    外层重新 iter()，epoch 计数 +1 触发重洗）。
    """

    def __init__(self, dataset: Dataset, seed: int = 0):
        self._dataset = dataset
        self._seed = seed
        self._epoch = 0

    def __iter__(self):
        rng = random.Random(self._seed + self._epoch)
        indices = list(range(len(self._dataset)))
        rng.shuffle(indices)
        self._epoch += 1
        for i in indices:
            yield self._dataset[i]

    def state_dict(self) -> dict:
        return {"epoch": self._epoch}

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state["epoch"]
```

---

## 7. 数据集实现

### 7.1 ChatDataset（对话格式数据集）

> **调用位置**: 时序树 ⑧.4 — `build_dataloader()` Step 4
> **参考**: Automodel `nemo_automodel/components/datasets/llm/chat_dataset.py`

#### 7.1.1 设计理念

ChatDataset 是对话格式 SFT 数据集的核心实现。支持：
- 从 HuggingFace Hub、本地 JSON/JSONL、Parquet 文件加载
- ShareGPT 格式（`conversations` 字段）和标准格式（`messages` 字段）自动转换
- 通过 `format_chat_template` 构建只对 assistant 回复计算 loss 的 labels
- 支持 `mask_history`（只保留最后一轮 assistant 回复的 loss）
- 支持 `mask_reasoning_content`（屏蔽 reasoning 内容 loss）

#### 7.1.2 配置

```python
# hyper_models/components/datasets/llm/chat_dataset.py

from dataclasses import dataclass
from typing import Optional, Union, Sequence, Any


@dataclass
class ChatDatasetConfig:
    """ChatDataset 配置。"""
    path_or_dataset_id: str | Sequence[str]
    split: str | None = None
    name: str | None = None
    seq_length: int | None = None
    padding: str | bool = "do_not_pad"
    truncation: str | bool = "do_not_truncate"
    start_of_turn_token: str | None = None
    chat_template: str | None = None
    shuffle_seed: int | None = None
    mask_reasoning_content: bool = False
    mask_history: bool = False
    unshifted: bool = False
    skip_invalid_samples: bool = False

    def build(self, *, tokenizer) -> "ChatDataset":
        """构建 ChatDataset 实例。"""
        return ChatDataset(
            path_or_dataset_id=self.path_or_dataset_id,
            tokenizer=tokenizer,
            split=self.split,
            name=self.name,
            seq_length=self.seq_length,
            padding=self.padding,
            truncation=self.truncation,
            start_of_turn_token=self.start_of_turn_token,
            chat_template=self.chat_template,
            shuffle_seed=self.shuffle_seed,
            mask_reasoning_content=self.mask_reasoning_content,
            mask_history=self.mask_history,
            unshifted=self.unshifted,
            skip_invalid_samples=self.skip_invalid_samples,
        )
```

#### 7.1.3 ChatDataset 实现

```python
class ChatDataset(torch.utils.data.Dataset):
    """对话格式 SFT 数据集。

    与 Automodel ChatDataset 对齐的完整实现。
    处理流程：
    1. 从 HF Hub / 本地 JSON 加载原始数据
    2. 统一消息格式（ShareGPT、标准格式）
    3. 调用 format_chat_template 构建 input_ids + labels
    4. labels 中非 assistant 回复位置设为 -100
    """

    def __init__(
        self,
        path_or_dataset_id: Union[str, Sequence[str]],
        tokenizer,
        *,
        split: Optional[str] = None,
        name: Optional[str] = None,
        seq_length: Optional[int] = None,
        padding: Union[str, bool] = "do_not_pad",
        truncation: Union[str, bool] = "do_not_truncate",
        start_of_turn_token: Optional[str] = None,
        chat_template: Optional[str] = None,
        shuffle_seed: Optional[int] = None,
        mask_reasoning_content: bool = False,
        mask_history: bool = False,
        unshifted: bool = False,
        skip_invalid_samples: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.padding = padding
        self.truncation = truncation
        self.mask_history = mask_history
        self.mask_reasoning_content = mask_reasoning_content
        self.unshifted = unshifted
        self.skip_invalid_samples = skip_invalid_samples

        # 加载数据
        self.dataset = self._load_openai_messages(
            path_or_dataset_id, split=split, name=name,
            shuffle_seed=shuffle_seed,
            skip_invalid_samples=skip_invalid_samples,
        )

        # 确保 pad_token 存在
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 设置 chat_template
        if chat_template is not None:
            self._resolve_chat_template(chat_template, tokenizer)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        """获取第 idx 个样本，返回 tokenized 结果。

        Returns:
            dict with keys: input_ids, labels, attention_mask
        """
        row = self.dataset[idx]

        # 兼容 ShareGPT 格式
        if "conversations" in row and "messages" not in row:
            row["messages"] = self._conversations_to_messages(row["conversations"])

        if "messages" not in row:
            raise KeyError(f"Sample at index {idx} has no 'messages' or 'conversations' field")

        messages = row["messages"]
        if not isinstance(messages, list):
            raise TypeError(f"Expected 'messages' to be a list, got {type(messages)}")

        # 标准化消息格式
        normalized = self._normalize_messages(messages)

        # 解析 tools（JSONL 格式常见）
        tools = None
        if "tools" in row:
            if isinstance(row["tools"], str):
                import json
                try:
                    tools = json.loads(row["tools"])
                except json.JSONDecodeError:
                    tools = None
            else:
                tools = row["tools"]

        # 调用 format_chat_template 构建 input_ids + labels
        from hyper_models.components.datasets.llm.formatting_utils import format_chat_template

        result = format_chat_template(
            tokenizer=self.tokenizer,
            formatted_text=normalized,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id or 0,
            seq_length=self.seq_length,
            padding=self.padding,
            truncation=self.truncation,
            tools=tools,
            answer_only_loss_mask=True,
            mask_reasoning_content=self.mask_reasoning_content,
            unshifted=self.unshifted,
        )

        # mask_history: 只保留最后一轮 assistant 回复的 loss
        if self.mask_history and not self.unshifted:
            from hyper_models.components.datasets.llm.formatting_utils import (
                _mask_labels_to_last_turn,
            )
            result["labels"] = _mask_labels_to_last_turn(result["labels"], ignore_index=-100)

        return result

    def __len__(self) -> int:
        return len(self.dataset)

    def _normalize_messages(self, messages: list[dict]) -> list[dict]:
        """标准化消息格式。

        与 Automodel ChatDataset._normalize_messages 对齐：
        - 验证 role 合法性
        - 标准化 content（list of parts → 纯文本）
        - 标准化 reasoning_content 和 tool_calls
        """
        valid_roles = {"system", "user", "assistant", "tool"}
        normalized = []

        for msg in messages:
            role = msg.get("role", "")
            if role not in valid_roles:
                continue

            entry = {"role": role}

            # 标准化 content
            content = msg.get("content", "")
            if isinstance(content, list):
                # list of parts: 只保留 text 部分
                texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = " ".join(texts) if texts else ""
            elif content is None:
                content = ""
            else:
                content = str(content)
            entry["content"] = content

            # assistant 特定字段
            if role == "assistant":
                reasoning = msg.get("reasoning_content", None)
                if reasoning is not None:
                    entry["reasoning_content"] = str(reasoning) if reasoning else ""

                tool_calls = msg.get("tool_calls", None)
                if tool_calls:
                    valid_calls = []
                    for tc in tool_calls:
                        if isinstance(tc, dict) and "function" in tc:
                            fn = tc["function"]
                            if isinstance(fn.get("arguments"), dict):
                                import json
                                fn = dict(fn)
                                fn["arguments"] = json.dumps(fn["arguments"], ensure_ascii=False)
                            valid_calls.append({**tc, "function": fn})
                    if valid_calls:
                        entry["tool_calls"] = valid_calls

            # tool 特定字段
            if role == "tool" and "tool_call_id" in msg:
                entry["tool_call_id"] = msg["tool_call_id"]

            normalized.append(entry)

        return normalized

    def _conversations_to_messages(self, conversations: list[dict]) -> list[dict]:
        """将 ShareGPT 格式转换为标准格式。

        ShareGPT: {"from": "human"/"gpt", "value": "..."}
        标准格式: {"role": "user"/"assistant", "content": "..."}
        """
        role_map = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}
        messages = []
        for conv in conversations:
            role = role_map.get(conv.get("from", ""), "user")
            messages.append({"role": role, "content": conv.get("value", "")})
        return messages

    def _load_openai_messages(
        self,
        path_or_dataset_id: Union[str, Sequence[str]],
        split: Optional[str] = None,
        name: Optional[str] = None,
        shuffle_seed: Optional[int] = None,
        skip_invalid_samples: bool = False,
    ):
        """加载数据集。

        与 Automodel ChatDataset._load_openai_messages 对齐：
        - HF Hub 路径 → load_dataset()
        - 本地 JSON/JSONL 文件 → 手动解析
        - 本地 Parquet 文件 → load_dataset() 读取
        """
        from datasets import load_dataset, Dataset as HFDataset
        import os, json

        # 多路径
        if isinstance(path_or_dataset_id, (list, tuple)):
            datasets = []
            for p in path_or_dataset_id:
                ds = self._load_openai_messages(
                    p, split=split, name=name,
                    skip_invalid_samples=skip_invalid_samples,
                )
                if isinstance(ds, HFDataset):
                    datasets.append(ds)
                elif isinstance(ds, list):
                    datasets.extend(ds)
            if all(isinstance(d, HFDataset) for d in datasets):
                from datasets import concatenate_datasets
                return concatenate_datasets(datasets)
            return [item for d in datasets for item in (d if isinstance(d, list) else [d])]

        # HF Hub 路径
        if isinstance(path_or_dataset_id, str) and "/" in path_or_dataset_id and not os.path.exists(path_or_dataset_id):
            return load_dataset(
                path_or_dataset_id, name=name, split=split,
                streaming=False,
            )

        # 本地文件
        if os.path.isdir(path_or_dataset_id) or path_or_dataset_id.endswith(".parquet"):
            return load_dataset(
                path_or_dataset_id, name=name, split=split,
                streaming=False,
            )

        # 本地 JSON/JSONL
        data = []
        if path_or_dataset_id.endswith(".json"):
            with open(path_or_dataset_id, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and split in data:
                    data = data[split]
        elif path_or_dataset_id.endswith(".jsonl"):
            with open(path_or_dataset_id, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        if not skip_invalid_samples:
                            raise
        else:
            raise ValueError(f"Unsupported dataset path: {path_or_dataset_id}")

        return data

    def _resolve_chat_template(self, chat_template: str, tokenizer):
        """解析 chat_template 名称并设置到 tokenizer。"""
        from hyper_models.components.datasets.llm.chat_templates import (
            CHAT_TEMPLATE_REGISTRY,
        )
        if chat_template in CHAT_TEMPLATE_REGISTRY:
            tokenizer.chat_template = CHAT_TEMPLATE_REGISTRY[chat_template]
        else:
            tokenizer.chat_template = chat_template
```

### 7.2 MegatronPretraining

> **调用位置**: 时序树 ⑧.4 — `build_dataloader()` Step 4（`_target_ == MegatronPretraining` 分支）

`MegatronPretraining` 是 Megatron `.bin/.idx` 二进制格式数据集的 `_target_` IoC 包装器。其 `.build()` 方法内部复用本地 `GPTDataset`/`BlendableDataset`/`indexed_dataset` 完成二进制索引加载与样本切分。

核心职责：
- 接收 `paths: list[str]`（支持 `"weight path"` 格式的多源混合）
- 通过 `GPTDataset` 按 `seq_length` 切分样本
- 通过 `BlendableDataset` 按权重混合多个子集
- `.build()` 完成 blend index 构建（副作用：设置内部 dataset 引用）
- `.get_dataset(split)` 返回实际可迭代的 `GPTDataset`/`BlendableDataset` 实例

配置示例：
```yaml
dataset:
  _target_: hyper_models.components.datasets.llm.MegatronPretraining
  paths:
    - "0.8 /data/corpus_a"
    - "0.2 /data/corpus_b"
  seq_length: 2048
```

> **注意**：Step 4 中 `ds = dataset_target(**kwargs); ds.build()`（`dataset_target` 为
> `_target_` 经 `import_target()` 解析后的 callable，解析入口 resolve_data_config 规划中）后 `ds` 仍为
> `MegatronPretraining` 实例（`.build()` 是副作用方法不改变类型）；Step 8 中通过
> `ds.get_dataset(split)` 获取底层 `GPTDataset`/`BlendableDataset` 用于 sampler。

#### 参考源码

| 源 | 路径 | 关键类/函数 |
|----|------|-----------|
| **AutoModel** | `auto_model/Automodel/nemo_automodel/components/datasets/llm/megatron_dataset.py` | `MegatronPretraining` — `__init__`(paths, seq_length, global_batch_size, micro_batch_size, seed, …), `build()`, `get_dataset(split)` |
| **本地（保留）** | `hyper_parallel/data/megatron/gpt_dataset.py` | `GPTDataset` — 按 `seq_length` 从 `.bin/.idx` 切分样本，含 `__getitem__`/`__len__` |
| **本地（保留）** | `hyper_parallel/data/megatron/blendable_dataset.py` | `BlendableDataset` — 按权重混合多个 `GPTDataset`，含 `state_dict`/`load_state_dict` |
| **本地（保留）** | `hyper_parallel/data/megatron/indexed_dataset.py` | `IndexedDataset` / `MMapIndexedDataset` — 二进制 `.bin/.idx` 索引加载 |
| **本地（保留）** | `hyper_parallel/data/megatron/builder.py` | `build_megatron()` — 函数式构建入口 |

**开发指引**：
1. 阅读本地 `gpt_dataset.py::GPTDataset` 了解样本切分逻辑（每个 `.bin` 文件被切为 `(file_tokens - 1) // seq_length` 个样本）
2. 阅读本地 `blendable_dataset.py::BlendableDataset` 了解多源混合（维护 `dataset_index` + `sample_index`，支持 `state_dict`/`load_state_dict`）
3. 参考 AutoModel `megatron_dataset.py::MegatronPretraining` 了解 `_target_` IoC 包装模式（`__init__` 接收训练参数 → `build()` 创建 `BlendableDataset`）
4. hyper_parallel 版 `MegatronPretraining` 放在 `hyper_models/components/datasets/llm/megatron_dataset.py`，导入本地 `hyper_parallel/data/megatron/` 模块

### 7.3 HF 数据集

> 直接使用 `datasets.load_dataset()` 实例化，无封装层。

---

## 8. IterableDataset 分片

> **调用位置**: 时序树 ⑧.5 — Step 6

分布式训练中，每个 DP rank 只应处理数据集的一个分片（shard）。分片策略取决于数据集类型：

**1. 带 `.shard()` 方法的 IterableDataset（如 HuggingFace `datasets.IterableDataset`）**：
直接调用 `ds.shard(num_shards=dp_world_size, index=dp_rank)`。

**2. HF streaming dataset（`ds.dataset` 为底层 `IterableDataset`）**：
使用 `datasets.distributed.split_dataset_by_node(ds.dataset, world_size, rank)`。

**3. map-style Dataset**：
不分片——由 Sampler（§10）通过 `StatefulDistributedSampler` 按 rank 分配索引。

实现见 §3.2 Step 6（`build_dataloader()` 中的 IterableDataset 分片分支）。

#### 参考源码

| 源 | 路径 | 关键函数 |
|----|------|---------|
| **HF datasets** | `datasets/distributed.py` | `split_dataset_by_node(dataset, world_size, rank)` — 按 worker group 分片 streaming dataset；内部计算 `per_node_shard = total_shards // num_nodes`，每个 node 取其对应的 `[rank*shards, (rank+1)*shards)` |
| **HF datasets** | `datasets/iterable_dataset.py` | `IterableDataset.shard(num_shards, index)` — 按 `index % num_shards` 取第 N 个元素，等价于 `islice(dataset, index, None, num_shards)` |
| **实现**（本文档） | `hyper_models/components/datasets/llm/dataloader.py` | `build_dataloader()` Step 6（第 337-345 行伪代码） |

**开发指引**：
1. HF streaming 场景：确保 `rank` 和 `world_size` 为 DP 维的 rank/size（非全局 rank），否则不同 DP rank 会拿到重复数据
2. 非 streaming IterableDataset：优先调用 `ds.shard()`（HF datasets API 标准方法）；若不存在该方法，检查 `ds.dataset` 是否可 shard
3. map-style Dataset 不需要此步——后续 Sampler 处理 DP 分配

---

## 9. Packed Sequence

> **调用位置**: 时序树 ⑧.6 — Step 7

将多个短序列打包为一个长序列，减少 padding 浪费。支持两种策略：

| 策略 | Packing 算法 | Collate 函数 | 适用场景 |
|------|------------|-------------|---------|
| **THD** (`packing_strategy="thd"`) | `pack_dataset()` — 贪心装箱 + cross-entropy ignore index 填充 | `packed_sequence_thd_collater` — 产出 `seq_lens`/`seq_lens_padded` + `qkv_format="thd"` | 纯文本 LLM 训练 |
| **NEAT** (`packing_strategy="neat"`) | `neat_pack_dataset()` — 支持 VLM 多模态打包 | `neat_packed_collater` — 需要 attention implementation 参数 | VLM 训练、跨模态打包 |

**配置字段**（来自 `PackingConfig` / `cfg_ps`）：
- `packed_sequence_size`: 打包后序列长度（0 = 不启用）
- `packing_strategy`: `"thd"` | `"neat"`
- `prepacked`: 数据集是否已预打包（True 时跳过 recipe-side packing）
- `max_packs`: 最大 pack 数量上限
- `drop_long_samples`（仅 NEAT）: 是否丢弃超过 pack_size 的超长样本

**模型兼容性检查**：THD packing 要求模型 `forward()` 接受 `seq_lens` 参数；
若 `_supports_seq_lens(model)` 为 False，自动降级关闭 packing 并 warn。

**CP 集成**：当 `cp_size > 1` 时，packing 按 `cp_size` 对齐 pack 中的样本数，
确保每个 CP rank 获得完整的 pack（而非部分 pack）。

实现见 §3.2 Step 7（`build_dataloader()` 中的 Packed Sequence 分支）。

#### 参考源码

| 源 | 路径 | 关键函数/类 |
|----|------|-----------|
| **AutoModel（THD）** | `auto_model/Automodel/nemo_automodel/components/datasets/llm/packed_sequence.py` | `pack_dataset()` — 主入口；`_fill_labels_with_cross_entropy_ignore_idx()` — loss mask 填充；`_should_stop_packing()` — 早停判断；`_split_and_add_pack()` — 分 pack + 添加；`_tensorize_and_pad_pack()` — tensor 化 + padding |
| **AutoModel（NEAT）** | `auto_model/Automodel/nemo_automodel/components/datasets/llm/neat_packing.py` | `neat_pack_dataset()` — VLM NEAT 打包入口 |
| **AutoModel（packing 配置）** | `auto_model/Automodel/nemo_automodel/components/models/common/packing.py` | `configure_packing(attn_implementation)` — 设置进程级 `_PACKING_CONFIG`；`get_attn_implementation(cfg_model)` — 推断 attention implementation |
| **实现**（本文档） | `hyper_models/components/datasets/llm/packed_sequence.py` | `pack_dataset()` — 对齐 AutoModel THD 实现 |
| **实现**（本文档） | `hyper_models/components/datasets/llm/neat_packing.py` | `neat_pack_dataset()` — 对齐 AutoModel NEAT 实现 |
| **实现**（本文档） | `hyper_models/components/models/common/packing.py` | `configure_packing()` / `get_attn_implementation()` — 对齐 AutoModel |

**开发指引**：
1. **THD packing 核心算法**：遍历样本，贪心尝试放入当前 pack；若放不下则 `_split_and_add_pack` 结束当前 pack 并开始新 pack。pack 内用 `-100`（`IGNORE_INDEX`）填充未使用位置，CE 自动忽略
2. **`seq_lens` 生成**：`_tensorize_and_pad_pack` 为每个 pack 生成 `seq_lens`（各样本实际长度）+ `seq_lens_padded`（对齐到 `max_num_packs`，用 `-1000` 哨兵）
3. **NEAT 与 THD 的关键差异**：NEAT 不修改 `input_ids`（保留原始 tokens），而是在 attention mask 层标记哪些 token 属于同一样本（通过 `cu_seqlens` 等方式传递）；THD 直接用 `-100` 填充未使用位置
4. **CP 对齐**：`cp_size > 1` 时，`pack_dataset` 接收 `cp_size` 参数，确保每个 pack 内样本数为 `cp_size` 的倍数；`_tensorize_and_pad_pack` 中对 sample boundary 做 cp_size 对齐

---

## 10. Sampler

> **调用位置**: 时序树 ⑧.7 — Step 8

Sampler 决定每个 DP rank 处理哪些样本索引。三条路径：

**路径 A — Megatron 数据集**：
`create_megatron_sampler(dataset_len, micro_batch_size, global_batch_size, dataloader_type, rank, world_size)` → `BatchSampler`。支持两种类型：
- `"single"`: `MegatronPretrainingSampler` — 每个 batch 取连续 `micro_batch_size` 个样本
- `"random"`: `MegatronPretrainingRandomSampler` — 随机采样

返回 `{"batch_sampler": batch_sampler}`，DataLoader 不使用 `batch_size`/`sampler`/`shuffle`。

**路径 B — map-style Dataset**：
`StatefulDistributedSampler(ds, seed, drop_last, num_replicas, rank, shuffle)`。
支持 checkpoint 续训状态追踪（`state_dict`/`load_state_dict`）。
当 `group_by_length=True` 时使用 `LengthGroupedSampler` 替代。

**路径 C — IterableDataset**：
无需 sampler（分片已在 §8 完成）。可选 `ds.shuffle(buffer_size, seed)` 进行流式 shuffle。

实现见 §3.2 Step 8（`build_dataloader()` 中的 Sampler 分支）。

#### 参考源码

| 源 | 路径 | 关键类/函数 |
|----|------|-----------|
| **AutoModel** | `auto_model/Automodel/nemo_automodel/components/datasets/llm/megatron/sampler.py` | `MegatronPretrainingSampler` — 构建连续 `micro_batch_size` 的 batch 索引；`MegatronPretrainingRandomSampler` — 随机采样索引；`create_megatron_sampler(dataset_len, micro_batch_size, global_batch_size, dataloader_type, rank, world_size)` — 工厂函数 |
| **torchdata** | `torchdata/stateful_dataloader/sampler.py` | `StatefulDistributedSampler(dataset, seed, drop_last, num_replicas, rank, shuffle)` — 支持 `state_dict()`/`load_state_dict()` 的分布式采样器；`set_epoch(epoch)` — 设置 shuffle 种子 |
| **实现**（本文档） | `hyper_models/components/datasets/llm/megatron/sampler.py` | 对齐 AutoModel 的 sampler 实现 |
| **实现**（本文档） | `hyper_models/components/datasets/llm/length_grouped_sampler.py` | `LengthGroupedSampler(dataset, batch_size, seed, num_replicas, rank)` — 按长度分组减少 padding |

**开发指引**：
1. **Megatron sampler 的关键逻辑**：`create_megatron_sampler` 返回 `BatchSampler`（而非 `Sampler`），每个 batch 为 `[start_idx, start_idx+1, ..., start_idx+micro_batch_size-1]`；DP rank 之间通过 `rank → rank+world_size → rank+2*world_size → ...` 交叉分配 batch
2. **`dataloader_type` 差异**：`"single"` 确保单个文档的样本不被拆分到不同 batch；`"random"` 全局随机索引
3. **`StatefulDistributedSampler` 状态**：`state_dict()` 返回 `{"seed": int, "epoch": int}`，`load_state_dict()` 恢复后调用 `set_epoch(epoch)` 重置 shuffle 种子
4. **`LengthGroupedSampler`**：按序列长度排序后分组，每组内 batch 长度相近，减少 padding 浪费；仅 map-style dataset 支持

---

## 11. Collate 函数

### 11.1 标准 LM Collate

> `default_collater` 实现：per-key padding + position_ids 自动生成 + seq_divisor 对齐

```python
# hyper_models/components/datasets/utils.py

def default_collater(tokenizer, seq_divisor: int = 1):
    """对已有 labels 做 per-key padding（labels 由 dataset 提供，本函数不生成）。

    tokenizer 可为 None（tokenizer 构建路径 2：YAML 显式 tokenizer: null）——
    此时回退 pad_token_id=0，仅作为兜底；需要正确 padding 语义时应配置 tokenizer。
    """
    pad_token_id = (getattr(tokenizer, "pad_token_id", None) or 0)

    ___PAD_TOKEN_IDS___ = {
        "input_ids": pad_token_id,
        "labels": -100,
        "attention_mask": 0,
        "position_ids": 0,
    }

    def collate(batch: list[dict]) -> dict:
        max_len = max(len(item["input_ids"]) for item in batch)

        if seq_divisor > 1:
            remainder = max_len % seq_divisor
            if remainder:
                max_len += seq_divisor - remainder

        if "position_ids" not in batch[0]:
            for item in batch:
                item["position_ids"] = torch.arange(len(item["input_ids"]))

        result = {}
        for key in ["input_ids", "labels", "attention_mask", "position_ids"]:
            if key not in batch[0]:
                continue
            pad_value = ___PAD_TOKEN_IDS___.get(key, 0)
            tensors = []
            for item in batch:
                vals = item[key]
                if not isinstance(vals, torch.Tensor):
                    vals = torch.as_tensor(vals)
                pad_len = max_len - len(vals)
                tensors.append(
                    torch.cat([vals, torch.full((pad_len,), pad_value, dtype=vals.dtype)])
                )
            result[key] = torch.stack(tensors)

        return result

    return collate
```

### 11.2 Packed Sequence Collate（THD）

> **调用位置**: 时序树 ⑧.8 — Step 9（cfg_dl.collate_fn 配置为 `packed_sequence_thd_collater`）

THD collate 处理 `pack_dataset()` 产出的 packed 样本。每个样本已包含 `input_ids`/`labels`/`position_ids`/`seq_lens`/`seq_lens_padded`（形状 `[pack_size]`），collate 只需 batchify：

1. 将 N 个 pack 的 `seq_lens_padded` 对齐到最大 pack 数（`max_num_packs`）
2. 对 `seq_lens`/`seq_lens_padded` 做 `pad_within_micro()`（填充哨兵 `-1000`）
3. 添加 `qkv_format = "thd"` 标识，供模型侧 varlen attention 识别

产出 batch dict：
```python
{
    "input_ids":          Tensor[B, pack_size],
    "labels":             Tensor[B, pack_size],
    "position_ids":       Tensor[B, pack_size],
    "seq_lens":           Tensor[B, max_num_packs],        # -1000 哨兵填充
    "seq_lens_padded":    Tensor[B, max_num_packs],        # 同上
    "qkv_format":         "thd",
}
```

实现见 `hyper_models/components/datasets/utils.py::packed_sequence_thd_collater`（helper: `get_pad_token_from_key`/`batchify`/`pad_within_micro`，签名见 §19）。

#### 参考源码

| 源 | 路径 | 关键函数 |
|----|------|---------|
| **AutoModel** | `auto_model/Automodel/nemo_automodel/components/datasets/utils.py` | `packed_sequence_thd_collater(batch)` — 主 collate 函数 |
| **AutoModel** | 同上 | `get_pad_token_from_key(key, pad_token_ids)` — 为 key 确定 pad 值（input_ids→pad_token_id, labels→-100, attention_mask→0, 其他→0） |
| **AutoModel** | 同上 | `batchify(tensor)` — 若 batch 中已有该 key，则按 dim-0 cat，否则 `torch.stack` |
| **AutoModel** | 同上 | `pad_within_micro(seq_lens_list, pad_value)` — 将 `seq_lens`/`seq_lens_padded` 对齐到 `max_num_packs` |
| **实现**（本文档） | `hyper_models/components/datasets/utils.py` | 对齐 AutoModel 的 `packed_sequence_thd_collater`、`batchify`、`pad_within_micro`、`get_pad_token_from_key` |
| **实现**（本文档） | 同上（§19 helper 签名） | `_fill_labels_with_cross_entropy_ignore_idx`、`_should_stop_packing`、`_split_and_add_pack`、`_tensorize_and_pad_pack` |

**开发指引**：
1. **`packed_sequence_thd_collater` 核心流程**：
   - 取 `max_num_packs = max(p["seq_lens_padded"].shape[0] for p in batch)`
   - 对每个 pack 的 `seq_lens`/`seq_lens_padded` 调用 `pad_within_micro`，不足 `max_num_packs` 的部分用 `-1000` 填充
   - 对 `input_ids`/`labels`/`position_ids` 做 `batchify`（普通 stack）
2. **`-1000` 哨兵**：模型中 `varlen` attention 通过检测 `seq_lens == -1000` 来忽略无效 pack 索引
3. **`qkv_format = "thd"`**：模型 forward 通过这个标识切换到 THD varlen attention 路径（而非标准 `[B, S]` attention）
4. **NEAT collate 对比**：`neat_packed_collater(batch, attn_implementation)` 产出不同的 batch 结构（含 `cu_seqlens_q`/`cu_seqlens_k` 等），不设 `qkv_format`，而是通过 attention implementation 参数切换计算路径

### 11.3 SequenceParallelCollator（新增）

> **参考**: VeOmni `veomni/data/data_collator.py` — `SequenceParallelCollator`

#### 11.3.1 设计理念

在 SP（Sequence Parallel）场景下，PackingCollator 完成打包后，需要沿 SP 维度对每个 key 做填充和切片，确保序列长度对 SP 划分均匀。SP collator 是 collate 管线中的最后一步，只在 `sp_enabled=True` 时启用。

#### 11.3.2 实现

```python
# hyper_models/components/datasets/utils.py

from dataclasses import dataclass


@dataclass
class DataCollateInfo:
    """每个字段的 collate 配置。

    pack_dim: 0 = cat along dim 0; -1 = cat along last dim then unsqueeze(0)
    sp_slice: 是否在 SP 切片
    sp_pad_value: SP padding 值（None = 不 pad）
    sp_pad_scale: SP padding 对齐尺度（如 pixel_values 的 ViT patch size=4）
    """
    pack_dim: int = -1
    sp_slice: bool = True
    sp_pad_value: int = 0
    sp_pad_scale: int = 1


DEFAULT_COLLATE_INFO = {
    "input_ids":         DataCollateInfo(-1, True, 0, 1),
    "labels":            DataCollateInfo(-1, True, -100, 1),
    # attention_mask pad 1 是 VeOmni 的刻意设计（已对照 veomni/data/
    # data_collator.py 核实）：SP 只切 input_ids/labels，保留全量
    # attention_mask 会导致 transformers create_causal_mask 切片错位；
    # 因此 VeOmni 保证 flash_attn 路径下 attention_mask 全 1，改由
    # position_ids / cu_seqlens / max_seqlen 预计算表达序列边界
    "attention_mask":    DataCollateInfo(-1, False, 1, 1),
    "position_ids":      DataCollateInfo(-1, False, 0, 1),
    "pixel_values":      DataCollateInfo(0, True, 0, 4),
    "pixel_values_videos": DataCollateInfo(0, True, 0, 4),
    "image_mask":        DataCollateInfo(-1, False, 0, 1),
    "video_mask":        DataCollateInfo(-1, False, 0, 1),
    "image_grid_thw":    DataCollateInfo(0, False, None, None),
    "video_grid_thw":    DataCollateInfo(0, False, None, None),
}


def sequence_parallel_collate(
    batch: dict,
    sp_size: int,
    sp_rank: int,
    collate_info: dict[str, DataCollateInfo] = None,
) -> dict:
    """SP collate：沿 SP 维度对 batch 做填充和切片。

    与 VeOmni SequenceParallelCollator.__call__ 对齐（VeOmni 从
    get_parallel_state() 取 sp_size/sp_rank，此处改为显式参数）。

    流程：
    1. labels 移位（丢弃第一个 token，末尾补 -100）
    2. 对每个 key 做 SP padding（对齐到 sp_size * pad_scale 的倍数）
    3. 对需要切片的 key 按 sp_rank 取各自的 contiguous shard
       （第 [sp_rank*chunk, (sp_rank+1)*chunk) 段）
    4. position_ids 不随第 3 步切片，在最后单独按 sp_rank 切片

    Args:
        batch: PackingCollator 产出的 batch
        sp_size: Sequence Parallel size
        sp_rank: 当前进程的 SP rank（决定取哪一段 shard）
        collate_info: 每个字段的 collate 配置，默认使用 DEFAULT_COLLATE_INFO

    Returns:
        dict: SP 切片后的 batch
    """
    if collate_info is None:
        collate_info = DEFAULT_COLLATE_INFO

    # labels 移位
    if "labels" in batch:
        batch["labels"] = torch.cat([
            batch["labels"][..., 1:],
            torch.full_like(batch["labels"][..., :1], -100),
        ], dim=-1)

    vit_sp_pad = {}
    for key, info in collate_info.items():
        if key not in batch:
            continue

        # SP padding
        if info.sp_pad_value is not None:
            pre_len = batch[key].shape[info.pack_dim]
            pad_scale = info.sp_pad_scale * sp_size
            chunk_size = ((pre_len + pad_scale - 1) // pad_scale) * pad_scale
            pad_amount = chunk_size - pre_len
            if pad_amount > 0:
                pad_shape = list(batch[key].shape)
                pad_shape[info.pack_dim] = pad_amount
                batch[key] = torch.cat([
                    batch[key],
                    torch.full(pad_shape, info.sp_pad_value, dtype=batch[key].dtype),
                ], dim=info.pack_dim)
            vit_sp_pad[key] = chunk_size - pre_len

        # SP slicing：按 sp_rank 取本 rank 的 contiguous shard
        # （不能所有 rank 都取第 0 段）
        if info.sp_slice and key != "position_ids":
            slice_len = batch[key].shape[info.pack_dim] // sp_size
            batch[key] = batch[key].narrow(info.pack_dim, sp_rank * slice_len, slice_len)

    # position_ids 特殊处理：最后按 sp_rank 切片
    if "position_ids" in batch:
        slice_len = batch["position_ids"].shape[-1] // sp_size
        batch["position_ids"] = batch["position_ids"].narrow(-1, sp_rank * slice_len, slice_len)

    # 添加 flash attention 需要的 kwargs
    from hyper_models.components.datasets.utils import add_flash_attention_kwargs_from_position_ids
    batch = add_flash_attention_kwargs_from_position_ids(batch)

    return batch
```

#### 11.3.3 接入位置说明

`sequence_parallel_collate` 是 **VeOmni 式 SP**（sp_size > 1，与 TP 组合的
sequence parallel）的保留能力，**不在 `build_dataloader` 的 11 步主流程中
默认启用**。hyper_parallel 的序列切分主路径是 CP：03 §8 在训练循环内调用
`shard_batch_for_cp`（05 canonical，已落地于
`hyper_models/components/distributed/cp_utils.py`）完成按 rank 的 batch
切分。仅当模型启用与 CP 语义不同的 VeOmni 式 SP 时，才由 Recipe 在 collate
管线末尾显式接入本函数（传入 sp_size/sp_rank）；SP 与 CP 不应同时对同一
序列维生效。

---

## 12. 动态 Batching

> **调用位置**: 时序树 ⑧.10 — `build_dataloader()` Step 11（可选）
> **参考**: VeOmni `veomni/data/dynamic_batching.py`

### 12.1 设计理念

动态 Batching 基于 token 预算而非固定样本数来组 batch。核心优势：
- 变长序列训练时减少填充浪费
- 通过 token 预算热身（warmup）逐步增加 batch 大小
- 支持 `total` 和 `effective` 两种 token 计数模式

**与梯度累积的语义边界（重要）**：`StepScheduler.grad_acc_steps` 由
`global_batch_size / (local_batch_size * dp_world_size)` 按**样本数**推导
（03 §4）。动态 batching 下每个 micro-batch 的样本数由 token 预算动态
决定，样本数口径的 `global_batch_size` 失去意义。因此启用
`dynamic_batching` 时应满足 `global_batch_size == local_batch_size *
dp_world_size`（即 `grad_acc_steps == 1`），全局 batch 的语义改由
`dynamic_batching.n_token_per_iter`（每 micro-batch token 预算）表达；
`StepScheduler` 的整除校验在该配置下天然通过。需要"多 micro-batch 累积"
与动态 batching 组合时，应先明确 token 预算与累积步数的联合语义再实现，
当前版本不支持该组合。

### 12.2 实现

```python
# hyper_models/components/datasets/dynamic_batching.py

from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from collections import deque
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass
class DynamicBatchingConfig:
    """动态 batching 配置。"""
    enabled: bool = False
    n_token_per_iter: int = 0              # 每个 micro-batch 的 token 预算
    buffer_size: int = 10000               # 缓冲池大小
    bsz_warmup_steps: int = 0              # 热身步数
    bsz_warmup_init_mbtoken: int = 0       # 初始 token 预算（热身用）
    physical_token_cap: int = 0            # 物理 token 上限（0 表示不限制）
    count_mode: str = "total"              # "total" | "effective"


class DynBszBuffer:
    """基于 token 预算的 buffer，贪心选择样本组成 micro-batch。

    与 VeOmni DynBszBuffer 对齐。
    使用贪心首次适应（greedy first-fit）策略从 buffer 中选择样本，
    尽量填满 n_token_per_iter 的 token 预算。

    算法：
    1. 从 DataLoader 预取样本到 buffer（append）
    2. 调用 get_samples() 时，从 buffer 中遍历选择样本
    3. 选中的样本组成一个 micro-batch 返回
    4. 未选中的样本留在 buffer 中等待下次
    """

    def __init__(
        self,
        get_length_fn: Optional[Callable[[dict], int]] = None,
        get_physical_length_fn: Optional[Callable[[dict], int]] = None,
    ):
        self.get_length_fn = get_length_fn
        self.get_physical_length_fn = get_physical_length_fn

        self._buffer: list[dict] = []
        self._buffer_sample_lens: list[int] = []
        self._buffer_physical_lens: list[int] = []
        self._del_idxs: list[int] = []
        self._cur_idx: int = 0
        self._all_token_cnt: int = 0
        self._all_physical_token_cnt: int = 0

    def append(self, item: dict) -> None:
        """添加一个样本到 buffer。"""
        seq_len = self.get_length_fn(item) if self.get_length_fn else item.get("attention_mask", torch.tensor([0])).sum().item()
        physical_len = (
            self.get_physical_length_fn(item) if self.get_physical_length_fn
            else seq_len
        )

        self._buffer.append(item)
        self._buffer_sample_lens.append(seq_len)
        self._buffer_physical_lens.append(physical_len)
        self._all_token_cnt += seq_len
        self._all_physical_token_cnt += physical_len

    def get_samples(
        self,
        n_token_per_iter: int,
        force: bool = True,
        physical_token_cap: Optional[int] = None,
    ) -> list[dict]:
        """从 buffer 中选出一个 micro-batch。

        贪心首次适应策略：
        - 从 cur_idx 开始遍历 buffer
        - 跳过已标记删除的样本
        - 选择满足 token 预算的样本
        - 第一个样本强制选择（即使超出预算）

        Args:
            n_token_per_iter: 每个 micro-batch 的 token 预算
            force: 是否强制选择第一个样本（即使超出预算）
            physical_token_cap: 物理 token 上限

        Returns:
            选中的样本列表
        """
        if len(self._buffer) == 0:
            return []

        selected = []
        cum_seq_len = 0
        cum_physical_len = 0

        while self._cur_idx < len(self._buffer):
            if self._cur_idx in self._del_idxs:
                self._cur_idx += 1
                continue

            seq_len = self._buffer_sample_lens[self._cur_idx]
            physical_seq_len = self._buffer_physical_lens[self._cur_idx]

            fits_effective = seq_len <= n_token_per_iter - cum_seq_len
            fits_physical = (
                physical_token_cap is None
                or physical_seq_len <= physical_token_cap - cum_physical_len
            )
            first_forced = force and cum_seq_len == 0

            if first_forced or (fits_effective and fits_physical):
                cum_seq_len += seq_len
                cum_physical_len += physical_seq_len
                self._del_idxs.append(self._cur_idx)
                selected.append(self._buffer[self._cur_idx])

            self._cur_idx += 1

            # 达到 token 预算就停止
            if not first_forced and cum_seq_len >= n_token_per_iter:
                break

        return selected

    def flush(self) -> None:
        """清理已选中的样本，重置索引。"""
        self._cur_idx = 0

        # 减去已删除样本的 token 计数
        for idx in self._del_idxs:
            self._all_token_cnt -= self._buffer_sample_lens[idx]
            self._all_physical_token_cnt -= self._buffer_physical_lens[idx]

        # 过滤掉已删除的样本
        self._buffer = [
            item for i, item in enumerate(self._buffer)
            if i not in self._del_idxs
        ]
        self._buffer_sample_lens = [
            l for i, l in enumerate(self._buffer_sample_lens)
            if i not in self._del_idxs
        ]
        self._buffer_physical_lens = [
            l for i, l in enumerate(self._buffer_physical_lens)
            if i not in self._del_idxs
        ]
        self._del_idxs = []

    @property
    def all_token_cnt(self) -> int:
        return self._all_token_cnt

    @property
    def all_physical_token_cnt(self) -> int:
        return self._all_physical_token_cnt

    @property
    def size(self) -> int:
        return len(self._buffer)


class TokenBasedBatchIterator:
    """包装 DataLoader，按 token 预算产出 micro-batch。

    与 StepScheduler 的集成：
    - StepScheduler 每次迭代调用 __next__() 获取一个 micro-batch
    - TokenBasedBatchIterator 内部维护 DynBszBuffer
    - 每次从 buffer 中选择满足 token 预算的样本

    两种 token 计数模式：
    - total: 按 attention_mask.sum() 计数（物理 token）
    - effective: 按 labels != IGNORE_INDEX 计数（仅计算 loss 的 token）
    """

    IGNORE_INDEX = -100

    def __init__(self, dataloader, config: DynamicBatchingConfig):
        self.dataloader = dataloader
        self.config = config
        self._step = 0
        self._iterator = iter(dataloader)

        # 构建 token 计数函数
        if config.count_mode == "total":
            get_length_fn = lambda item: item.get("attention_mask", torch.ones_like(item["input_ids"])).sum().item()
        else:  # effective
            get_length_fn = lambda item: (
                (item.get("labels", item["input_ids"]) != self.IGNORE_INDEX).sum().item()
            )

        self.buffer = DynBszBuffer(
            get_length_fn=get_length_fn,
            get_physical_length_fn=None,
        )

    def __iter__(self):
        self._step = 0
        self._iterator = iter(self.dataloader)
        self.buffer = DynBszBuffer(
            get_length_fn=self.buffer.get_length_fn,
            get_physical_length_fn=self.buffer.get_physical_length_fn,
        )
        return self

    def __next__(self) -> dict:
        """返回一个 micro-batch。

        从 buffer 中选择满足 token 预算的样本。
        如果 buffer 不够，从 dataloader 继续取数据。

        架构约定（重要）：
        - 动态 batching 模式下，底层 DataLoader 应配置 batch_size=1 且不使用
          collate_fn（或将 collate_fn 设置为恒等函数），使得 `next(self._iterator)`
          返回单个样本 dict（shape [S]，非 batched [1, S]）。
        - TokenBasedBatchIterator 收集多个样本后，在自己的 collate 阶段（第 1920-1921
          行）统一调用底层 DataLoader 的 collate_fn 组装 batch。
        - 如果底层 DataLoader 已经 collate 过了（如 default_collater 产出 [1, S]
          tensor），需要在进入 buffer 前 squeeze batch 维度。
        """
        while True:
            # 尝试从 buffer 中弹出 micro-batch
            n_token = self._get_cur_token_budget()
            micro_batch = self.buffer.get_samples(
                n_token,
                force=True,
                physical_token_cap=self.config.physical_token_cap or None,
            )

            if micro_batch and len(micro_batch) > 0:
                self.buffer.flush()
                self._step += 1
                # 使用 collate_fn 组装 batch（buffer 中每条样本为单样本 dict）
                if hasattr(self.dataloader, "collate_fn"):
                    return self.dataloader.collate_fn(micro_batch)
                return micro_batch

            # buffer 不足，从 dataloader 取数据
            try:
                batch = next(self._iterator)
            except StopIteration:
                # 如果 buffer 还有数据且 force=True，强制输出
                if self.buffer.size > 0:
                    self.buffer.flush()
                    continue
                raise

            # 将 batch 中的每个样本加入 buffer。
            # 假设底层 DataLoader 产出单样本 dict（batch_size=1 且 collate_fn
            # 为恒等或已 squeeze），或产出 list[dict]（batch_size=N 且 collate_fn
            # 为恒等）。
            if isinstance(batch, dict):
                # 单样本 dict：检查是否需要 squeeze batch 维度
                # 若 tensor shape[0] == 1（已被 collate 为 [1, S]），squeeze
                _sample = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == 1:
                        _sample[k] = v.squeeze(0)
                    else:
                        _sample[k] = v
                self.buffer.append(_sample)
            elif isinstance(batch, (list, tuple)):
                for item in batch:
                    if isinstance(item, dict):
                        self.buffer.append(item)
            else:
                logger.warning("Unexpected batch type from DataLoader: %s; skipping", type(batch))

    def _get_cur_token_budget(self) -> int:
        """获取当前步的 token 预算（含热身）。"""
        base = self.config.n_token_per_iter
        if self.config.bsz_warmup_steps > 0 and self._step < self.config.bsz_warmup_steps:
            return (
                (base - self.config.bsz_warmup_init_mbtoken) * self._step
                // self.config.bsz_warmup_steps
                + self.config.bsz_warmup_init_mbtoken
            )
        return base

    def __len__(self) -> int:
        # 动态 batching 不支持 __len__
        raise TypeError("TokenBasedBatchIterator does not support __len__")

    def state_dict(self) -> dict:
        return {
            "step": self._step,
            "buffer_size": self.buffer.size,
            "config": self.config,
        }

    def load_state_dict(self, state: dict):
        self._step = state["step"]
```

---

## 13. BackgroundPrefetcher

> **参考**: VeOmni `veomni/trainer/base.py` — `BackgroundPrefetcher`

### 13.1 设计理念

BackgroundPrefetcher 在后台线程中预取 DataLoader 的下一个 batch，当 GPU 计算时，数据加载与计算重叠，提升训练吞吐量。同时捕获 DataLoader 的 `state_dict()` 用于 checkpoint 恢复。

### 13.2 实现

```python
# hyper_models/components/datasets/prefetch.py

import threading
import queue
from typing import Optional


class BackgroundPrefetcher:
    """后台线程预取 DataLoader 的 batch。

    使用方式：
        loader = BackgroundPrefetcher(dataloader, prefetch_size=2)
        for batch in loader:
            train_step(batch)

    与 VeOmni BackgroundPrefetcher 对齐：
    - 后台线程持续从 DataLoader 取数据
    - 通过 queue.Queue 传递 batch
    - 同时捕获 DataLoader 的 state_dict() 用于 checkpoint 恢复

    多 epoch 支持：底层迭代器耗尽后，下次 __iter__ 自动重启预取线程
    （StepScheduler 每个 epoch 重新迭代 dataloader，见 03 §4/§6）。
    """

    # 耗尽哨兵：不能用 StopIteration 类对象做标记——类不是自身的实例，
    # isinstance(item, StopIteration) 恒为 False，会被误当 batch 返回
    _STOP = object()

    def __init__(self, dataloader, prefetch_size: int = 2):
        self.dataloader = dataloader
        self.prefetch_size = prefetch_size
        self._original_state_dict = getattr(dataloader, "state_dict", None)
        self._current_state = None
        self._exhausted = False
        self._start()

    def _start(self):
        """（重）建底层迭代器与预取线程。"""
        self._iterator = iter(self.dataloader)
        self._queue: queue.Queue = queue.Queue(maxsize=self.prefetch_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._thread.start()
        self._exhausted = False

    def _prefetch_loop(self):
        """后台线程主循环。"""
        while not self._stop_event.is_set():
            try:
                item = next(self._iterator)
            except StopIteration:
                self._queue.put((self._STOP, None))
                break
            except Exception as e:
                self._queue.put((e, None))
                break

            # 捕获 DataLoader 状态
            state = self._original_state_dict() if self._original_state_dict else None
            self._queue.put((item, state))

    def __next__(self):
        item, state = self._queue.get()
        self._current_state = state

        if item is self._STOP:
            self._exhausted = True
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item

    def __iter__(self):
        # 多 epoch：上一轮已耗尽时重启底层迭代器与预取线程
        if self._exhausted:
            self._start()
        return self

    def state_dict(self):
        """返回最近的 DataLoader 状态。"""
        if self._current_state is not None:
            return self._current_state
        if self._original_state_dict:
            return self._original_state_dict()
        return {}

    def stop(self, timeout: float = 5.0):
        """停止后台线程。"""
        self._stop_event.set()
        # 清空队列以解除阻塞
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=timeout)
```

---

## 14. 辅助工具

### 14.1 ReservoirSampler

> **参考**: Automodel `nemo_automodel/components/datasets/reservoir_sampler.py`

```python
# hyper_models/components/datasets/reservoir_sampler.py

import random
from typing import Iterable, Iterator, Optional


class ReservoirSampler:
    """有界内存的流式 shuffle 缓冲。

    用于流式（Iterable）数据集的 shuffle：
    1. 填充 buffer 到 buffer_size
    2. 随机 evict 一个元素并 yield
    3. 从底层 iterator 取下一个元素补充
    4. 循环直到 iterator 耗尽

    与 Automodel ReservoirSampler 对齐。
    """

    def __init__(self, iterator: Iterable, buffer_size: int, seed: Optional[int] = None):
        self._buffer_size = int(buffer_size)
        self._seed = seed
        self._iterable = iterator

    def __iter__(self) -> Iterator:
        rng = random.Random(self._seed)

        # 填充 buffer
        buffer = []
        iterator = iter(self._iterable)
        for item in iterator:
            buffer.append(item)
            if len(buffer) >= self._buffer_size:
                break

        rng.shuffle(buffer)

        # 主循环：随机 evict + 补充
        for item in iterator:
            new_pos = rng.randint(0, len(buffer) - 1)
            evicted = buffer[new_pos]
            buffer[new_pos] = item
            yield evicted

        # 剩余 buffer 中的元素全部 yield
        rng.shuffle(buffer)
        yield from buffer
```

### 14.2 LazyMappedDataset

> **参考**: Automodel `nemo_automodel/components/datasets/lazy_mapped_dataset.py`

```python
# hyper_models/components/datasets/lazy_mapped_dataset.py

from functools import lru_cache
from torch.utils.data import Dataset


class LazyMappedDataset(Dataset):
    """延迟映射数据集：在 __getitem__ 时应用 map_fn，不提前预处理。

    与 Automodel LazyMappedDataset 对齐：
    - 延迟执行：__getitem__ 时实时应用 map_fn
    - LRU 缓存：cache_size > 0 时缓存最近访问的结果
    - Pickle 安全：__getstate__ 清理 _get_item，__setstate__ 重建

    Args:
        dataset: 底层数据集
        map_fn: 映射函数，接收原始样本返回变换后的样本
        cache_size: LRU 缓存大小（None 表示缓存所有，0 表示不缓存）
    """

    def __init__(self, dataset, map_fn, cache_size=10000):
        self._dataset = dataset
        self._map_fn = map_fn
        if cache_size is None:
            cache_size = len(dataset)
        self._cache_size = cache_size
        self._build_get_item()

    def _build_get_item(self):
        if self._cache_size > 0:
            @lru_cache(maxsize=self._cache_size)
            def _cached_transform(idx: int):
                return self._map_fn(self._dataset[idx])
            self._get_item = _cached_transform
        else:
            self._get_item = lambda idx: self._map_fn(self._dataset[idx])

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_get_item"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._build_get_item()

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        return self._get_item(idx)

    @property
    def cache_info(self):
        fn = self._get_item
        if hasattr(fn, "cache_info"):
            return fn.cache_info()
        return None
```

---

## 15. VLM 数据管道

### 15.1 VlmDataloaderConfig

> **参考**: Automodel `nemo_automodel/components/datasets/vlm/loader.py` — `VlmDataloaderConfig`

```python
# hyper_models/components/datasets/vlm/loader.py

import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

logger = logging.getLogger(__name__)


@dataclass
class VlmProcessorConfig:
    """VLM processor 配置。"""
    factory: Optional[Callable] = None
    kwargs: dict = field(default_factory=dict)

    def build(self, *, pretrained_model_name_or_path: str):
        """构建 processor。

        使用 AutoProcessor.from_pretrained 或自定义工厂函数。
        """
        if self.factory is not None:
            return self.factory(pretrained_model_name_or_path, **self.kwargs)
        from transformers import AutoProcessor
        try:
            return AutoProcessor.from_pretrained(
                pretrained_model_name_or_path, **self.kwargs
            )
        except Exception as e:
            logger.warning(f"Failed to load processor: {e}")
            return None


@dataclass(frozen=True)
class VlmDataloaderBuild:
    """VLM 数据加载器构建结果。"""
    dataloader: DataLoader
    processor: Any  # ProcessorMixin | None


@dataclass
class VlmDataloaderConfig:
    """VLM 数据加载器配置。

    与 Automodel VlmDataloaderConfig 对齐。
    与 LLM Dataloader 的差异：
    - 需要 processor（含 tokenizer + image processor）
    - 数据集使用 conversation 格式（含图像引用）
    - 可选 pretokenization（提前应用 processor）
    - collate 需处理 pixel_values / image_grid_thw
    """
    dataset_config: Any  # DatasetConfig
    processor_config: VlmProcessorConfig = field(default_factory=VlmProcessorConfig)
    pretokenization: Optional[Any] = None  # PreTokenizedDatasetWrapperConfig
    packing: Optional[Any] = None  # NeatPackConfig
    collator: Optional[Any] = None  # VlmCollatorConfig
    chat_template: Optional[str] = None
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = None
    drop_last: bool = False

    def build(
        self,
        *,
        pretrained_model_name_or_path: str,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int,
        packing_attn_implementation: Optional[str] = None,
        pp_n_microbatches: Optional[int] = None,
    ) -> VlmDataloaderBuild:
        """构建 VLM DataLoader。

        流程：
        1. 构建 processor（含 tokenizer + image processor）
        2. 构建数据集（conversation 格式）
        3. 可选 pretokenization（提前应用 processor）
        4. 可选 packing（THD 或 NEAT）
        5. 解析 collate 函数
        6. 构建 DistributedSampler 和 StatefulDataLoader
        """
        # 1. 构建 processor
        processor = self.processor_config.build(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
        )

        # 设置 chat_template
        if self.chat_template and processor is not None:
            tokenizer = getattr(processor, "tokenizer", processor)
            if hasattr(tokenizer, "chat_template"):
                tokenizer.chat_template = self.chat_template

        # 2. 构建数据集（注：ConfigNode 已不存在，直接通过 typed config.build() 构造）
        tokenizer = getattr(processor, "tokenizer", processor) if processor else None
        ds = self.dataset_config.build(tokenizer=tokenizer)

        # 3. 可选 pretokenization
        if self.pretokenization is not None:
            from hyper_models.components.datasets.vlm.datasets import (
                PreTokenizedDatasetWrapper,
            )
            max_length = getattr(self.pretokenization, "max_length", 8192)
            ds = PreTokenizedDatasetWrapper(
                dataset=ds,
                processor=processor,
                max_length=max_length,
            )

        # 4. 可选 packing
        if self.packing is not None:
            from hyper_models.components.datasets.utils import (
                packed_sequence_thd_collater, neat_packed_collater,
            )
            padding_idx = getattr(tokenizer, "pad_token_id", 0)
            if self.packing.get("packing_format", "thd") == "thd":
                from hyper_models.components.datasets.llm.packed_sequence import pack_dataset
                ds = pack_dataset(
                    ds,
                    split=self.packing.get("split", "train"),
                    packed_sequence_size=self.packing.get("pack_size", 8192),
                    max_packs=self.packing.get("max_packs", None),
                    padding_idx=padding_idx,
                )
                collate_fn = lambda batch, pi=padding_idx: packed_sequence_thd_collater(batch)
            else:
                from hyper_models.components.datasets.llm.neat_packing import neat_pack_dataset
                ds = neat_pack_dataset(
                    ds,
                    split=self.packing.get("split", "train"),
                    pack_size=self.packing.get("pack_size", 8192),
                    max_packs=self.packing.get("max_packs", None),
                    padding_idx=padding_idx,
                    drop_long_samples=self.packing.get("drop_long_samples", True),
                )
                collate_fn = lambda batch, ai=packing_attn_implementation: neat_packed_collater(
                    batch, attn_implementation=ai or "sdpa",
                )
        else:
            # 5. 解析 collate
            if self.collator is not None:
                collate_fn = self.collator.build(processor=processor)
            elif processor is not None:
                from hyper_models.components.datasets.vlm.collate_fns import pad_collate_fn
                collate_fn = lambda batch, p=processor: pad_collate_fn(batch, processor=p)
            else:
                from hyper_models.components.datasets.utils import default_collater
                collate_fn = default_collater(tokenizer)

        # PP 模式下包装 collate
        if pp_n_microbatches is not None:
            from hyper_models.components.datasets.vlm.pp_media import (
                wrap_vlm_collate_for_pp,
            )
            collate_fn = wrap_vlm_collate_for_pp(collate_fn, pp_n_microbatches)

        # 6. 构建 StatefulDistributedSampler 和 DataLoader
        sampler = StatefulDistributedSampler(
            dataset=ds,
            num_replicas=dp_world_size,
            rank=dp_rank,
            shuffle=self.shuffle,
            seed=0,  # VLM sampler seed 由外部控制
        )

        dl = StatefulDataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=self.drop_last,
        )

        return VlmDataloaderBuild(dataloader=dl, processor=processor)
```

### 15.2 PreTokenizedDatasetWrapper

```python
# hyper_models/components/datasets/vlm/datasets.py

import random
import torch
from torch.utils.data import Dataset


class PreTokenizedDatasetWrapper(Dataset):
    """在 __getitem__ 中应用 processor 做 tokenization。

    与 Automodel PreTokenizedDatasetWrapper 对齐的关键行为：
    - 超长样本：随机替换（而非丢弃）
    - 纯文本样本：注入 fake image 避免 FSDP batch 结构不一致
    - labels：只对 assistant token 计算 loss
    """

    def __init__(self, dataset, processor, max_length=8192, max_retries=10, truncate=False):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length
        self.max_retries = max_retries
        self.truncate = truncate
        self._rng = random.Random(42)

    def __getitem__(self, idx):
        from hyper_models.components.datasets.vlm.media_utils import (
            _extract_media_from_conversations,
            _preload_media,
        )
        from hyper_models.components.datasets.vlm.collate_fns import (
            build_labels_from_template,
        )
        from hyper_models.components.datasets.vlm.fake_image import (
            _conversation_has_media,
            inject_fake_image_into_conversation,
        )

        for attempt in range(self.max_retries):
            example = self.dataset[idx]
            example = _preload_media(example, self.processor)

            conversation = example.get("conversation", example.get("messages", []))

            # 纯文本样本注入 fake image（FSDP 兼容性）
            if not _conversation_has_media(conversation):
                conversation = inject_fake_image_into_conversation(conversation)

            # 应用 chat_template
            text = self.processor.apply_chat_template(conversation, tokenize=False)

            # 提取媒体
            images, videos = _extract_media_from_conversations(conversation)

            # 通过 processor 编码
            result = self.processor(
                text=[text],
                images=images if images else None,
                videos=videos if videos else None,
                return_tensors="pt",
                padding=True,
                truncation=self.truncate,
                max_length=self.max_length if self.truncate else None,
            )

            seq_len = result["input_ids"].shape[1]
            if self.max_length and seq_len > self.max_length and not self.truncate:
                idx = self._rng.randint(0, len(self.dataset) - 1)
                continue  # 重试

            # 构建 labels
            labels = build_labels_from_template(
                result["input_ids"], conversation, self.processor
            )
            result["labels"] = labels

            # 移除 batch 维度
            return {k: v.squeeze(0) if v.dim() > 1 and v.shape[0] == 1 else v
                    for k, v in result.items()}

        # 重试耗尽，返回最后一个结果（即使超长）
        return result

    def __len__(self):
        return len(self.dataset)
```

### 15.3 媒体工具

```python
# hyper_models/components/datasets/vlm/media_utils.py

import PIL.Image
from typing import Optional, Union


def smart_resize(
    image: PIL.Image,
    min_pixels: int = 56 * 56,
    max_pixels: int = 1280 * 28 * 28,
    resize_factor: int = 28,
) -> PIL.Image:
    """智能缩放：保持宽高比，限制像素数在 [min_pixels, max_pixels] 内，
    且宽高均为 resize_factor 的倍数。

    与 VeOmni multimodal/image_utils.py 的 smart_resize 对齐。
    """
    import math

    width, height = image.size
    pixels = width * height

    # 计算缩放比例
    if pixels < min_pixels:
        scale = math.sqrt(min_pixels / pixels)
    elif pixels > max_pixels:
        scale = math.sqrt(max_pixels / pixels)
    else:
        return image

    new_width = round(width * scale / resize_factor) * resize_factor
    new_height = round(height * scale / resize_factor) * resize_factor

    return image.resize((new_width, new_height), PIL.Image.LANCZOS)


def preload_image(source: Union[str, PIL.Image.Image, bytes]) -> PIL.Image.Image:
    """从路径、URL、bytes 或 PIL.Image 加载图像。"""
    if isinstance(source, PIL.Image.Image):
        return source
    if isinstance(source, bytes):
        from io import BytesIO
        return PIL.Image.open(BytesIO(source))
    if isinstance(source, str):
        if source.startswith(("http://", "https://")):
            import requests
            from io import BytesIO
            resp = requests.get(source, timeout=30)
            resp.raise_for_status()
            return PIL.Image.open(BytesIO(resp.content))
        return PIL.Image.open(source)
    raise TypeError(f"Unsupported image source type: {type(source)}")


def preload_video(
    source: str,
    max_frames: int = 32,
    fps: Optional[float] = None,
) -> list[PIL.Image.Image]:
    """使用 torchcodec 或 PyAV 解码视频，按帧率采样。"""
    try:
        import torchcodec
        decoder = torchcodec.Decoder(source)
        if fps:
            indices = [int(i * decoder.metadata.fps / fps)
                       for i in range(min(max_frames, int(decoder.metadata.fps / fps) if fps else max_frames))]
        else:
            indices = list(range(0, len(decoder), max(1, len(decoder) // max_frames)))
        frames = [decoder[i].data for i in indices]
        return [PIL.Image.fromarray(f.numpy()) for f in frames]
    except ImportError:
        from decord import VideoReader, cpu
        vr = VideoReader(source, ctx=cpu(0))
        if fps:
            frame_indices = list(range(0, len(vr), int(vr.get_avg_fps() / fps)))
        else:
            frame_indices = list(range(0, len(vr), max(1, len(vr) // max_frames)))
        return [PIL.Image.fromarray(vr[i].asnumpy()) for i in frame_indices[:max_frames]]


def preload_audio(source: str, target_sr: int = 16000):
    """使用 librosa 加载音频，重采样到 target_sr。"""
    import librosa
    import numpy as np
    audio, sr = librosa.load(source, sr=target_sr, mono=True)
    return audio, target_sr


def _extract_media_from_conversations(conversation: list[dict]):
    """从 conversation 中提取所有图像和视频。

    Returns:
        (images: list[PIL.Image], videos: list[list[PIL.Image]])
    """
    images = []
    videos = []
    for msg in conversation:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "image" and "image" in part:
                        images.append(preload_image(part["image"]))
                    elif part.get("type") == "video" and "video" in part:
                        videos.append(preload_video(part["video"]))
    return images, videos


def _preload_media(example: dict, processor) -> dict:
    """预加载 example 中的媒体资源。"""
    conversation = example.get("conversation", example.get("messages", []))
    for msg in conversation:
        content = msg.get("content", [])
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "image" and isinstance(part.get("image"), str):
                        part["image"] = preload_image(part["image"])
                    elif part.get("type") == "video" and isinstance(part.get("video"), str):
                        part["video"] = preload_video(part["video"])
    return example
```

### 15.4 VLM Collate 函数

```python
# hyper_models/components/datasets/vlm/collate_fns.py

import torch


def pad_collate_fn(batch: list[dict], processor=None) -> dict:
    """VLM 标准 padding collate。

    处理 input_ids、pixel_values、image_grid_thw 等 VLM 特有字段。
    """
    result = {}

    # 文本字段：padding
    for key in ["input_ids", "labels", "attention_mask"]:
        if key not in batch[0]:
            continue
        max_len = max(item[key].shape[-1] for item in batch)
        tensors = []
        for item in batch:
            pad_len = max_len - item[key].shape[-1]
            if pad_len > 0:
                pad_val = -100 if key == "labels" else 0
                tensors.append(torch.cat([
                    item[key],
                    torch.full((pad_len,), pad_val, dtype=item[key].dtype),
                ]))
            else:
                tensors.append(item[key])
        result[key] = torch.stack(tensors)

    # 图像字段：stack
    for key in ["pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"]:
        if key in batch[0] and batch[0][key] is not None:
            result[key] = torch.cat([item[key] for item in batch], dim=0)

    return result


def build_labels_from_template(
    input_ids: torch.Tensor,
    conversation: list[dict],
    processor,
) -> torch.Tensor:
    """根据 chat template 构建 labels：只对 assistant 回复计算 loss。"""
    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return labels
```

---

## 16. 配置示例

### 16.1 ChatDataset SFT 配置

```yaml
recipe: FinetuneRecipe

dataset:
  _target_: hyper_models.components.datasets.llm.ChatDataset
  path_or_dataset_id: HuggingFaceH4/ultrachat_200k
  split: train
  seq_length: 8192
  mask_history: true
  mask_reasoning_content: false

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  batch_size: 1
  num_workers: 2
  pin_memory: true
```

### 16.2 多源加权采样配置

```yaml
recipe: FinetuneRecipe

multisource:
  sources:
    - _target_: hyper_models.components.datasets.llm.ChatDataset
      path_or_dataset_id: HuggingFaceH4/ultrachat_200k
      split: train
      weight: 0.3
    - _target_: hyper_models.components.datasets.llm.MegatronPretraining
      paths: ["0.8 /data/corpus_a", "0.2 /data/corpus_b"]
      seq_length: 2048
      weight: 0.7
  seed: 1234
  level: sample
  stopping_strategy: never_exhausted

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  batch_size: 1
  num_workers: 2
  pin_memory: true
```

### 16.3 动态 Batching 配置

```yaml
recipe: FinetuneRecipe

dataset:
  _target_: hyper_models.components.datasets.llm.ChatDataset
  path_or_dataset_id: HuggingFaceH4/ultrachat_200k
  split: train
  seq_length: 8192

dynamic_batching:
  enabled: true
  n_token_per_iter: 2097152          # 256 * 8192（micro_batch_size * max_seq_len）
  buffer_size: 10000
  bsz_warmup_steps: 1000
  bsz_warmup_init_mbtoken: 200
  count_mode: total

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  batch_size: 1
  num_workers: 2
  pin_memory: true
```

### 16.4 Transform 注册表配置

```yaml
recipe: FinetuneRecipe

dataset:
  _target_: datasets.load_dataset
  path: HuggingFaceFW/fineweb
  name: sample-10BT
  split: train
  streaming: true

transform:
  name: plaintext
  text_keys: ["text"]
  cache_size: 10000

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  batch_size: 1
  num_workers: 2
  pin_memory: true
```

### 16.5 VLM 配置

```yaml
recipe: FinetuneRecipeForVLM

vlm_dataloader:
  dataset_config:
    path_or_dataset: HuggingFaceH4/llava-instruct
    split: train
  processor_config:
    factory: null
    kwargs:
      trust_remote_code: true
  pretokenization:
    max_length: 8192
  packing: null
  shuffle: true
  num_workers: 2
  pin_memory: true
```

---

## 17. 与 StepScheduler 的集成

训练循环中，StepScheduler 消费 `build_dataloader` 返回的 DataLoader。当启用了动态 Batching 时，StepScheduler 需要适配 TokenBasedBatchIterator 的接口。

```python
# 03_training_loop.md 中的集成示例

# 构建 DataLoader
# 注：cfg.dataset / cfg.dataloader / cfg.packed_sequence / cfg.dynamic_batching /
# cfg.transform / cfg.multisource 均为规划中的数据管道配置段，经 resolve_data_config
# 独立解析，不在当前 TrainerConfig 字段内（resolve_root() 拒绝未知一级字段）
self.dataloader, self.tokenizer = build_dataloader(
    cfg.dataset, cfg.dataloader, cfg.model, cfg.packed_sequence,
    cfg_db=cfg.dynamic_batching,  # 可选
    cfg_transform=cfg.transform,  # 可选
    cfg_multisource=cfg.multisource,  # 可选
    ...
)

# 可选 BackgroundPrefetcher 包装
if use_background_prefetcher:
    from hyper_models.components.datasets.prefetch import BackgroundPrefetcher
    self.dataloader = BackgroundPrefetcher(self.dataloader, prefetch_size=2)

# StepScheduler 消费
for epoch in self.step_scheduler.epochs:
    for batches in self.step_scheduler:
        # 当启用动态 Batching 时，TokenBasedBatchIterator.__next__()
        # 返回一个 collated micro-batch（由多个样本组成）
        # 当未启用时，StepScheduler 按固定 batch_size 取
        ...
```

---

## 18. 配置类型化解析层（规划中）

> **参考**: Automodel `nemo_automodel/recipes/_typed_config.py` — `_resolve_dataloader`

为提升配置的类型安全性和 IDE 支持，`hyper_models/components/datasets/config.py` 中使用强类型 dataclass 定义数据集配置。与 01 §2 的 `TrainerConfig` 保持一致，均基于 `Configurable.Config`（01 §3）或独立 dataclass。

> **定位说明（与 D4 统一）**：dataset/dataloader/packed_sequence/dynamic_batching/
> transform/multisource 等数据管道配置段不在当前 `TrainerConfig` 的 9 个一级字段内
> （`resolve_root()` 拒绝未知一级字段），因此不由 `resolve_component()` 解析，
> 而由规划中的独立入口 `resolve_data_config(cfg_dict)` 解析——机制复用 01 §2.4 的
> `import_target()` + `coerce_value()`，产出"已解析 callable + typed kwargs"或
> 直接产出本节定义的 Config 对象；嵌套 `_target_` 段（如 collate_fn、tokenizer、
> multisource.sources[*]）由 `resolve_data_config` 递归解析。

```python
# hyper_models/components/datasets/config.py

from dataclasses import dataclass, field
from typing import Optional, Any

from hyper_models.config.resolver import import_target


@dataclass
class DatasetConfig:
    """类型化数据集配置基类。

    `_target_` 声明要调用的数据集类/函数（字符串 dotted path），其余字段为其构造参数。
    """
    _target_: str
    split: Optional[str] = None

    def build(self, **kwargs) -> Any:
        """构建数据集实例——解析 `_target_` 并传入配置字段 + runtime kwargs。"""
        # _target_ 为字符串 dotted path，build() 内先经 import_target()（01 §2.4）
        # 解析为 callable 再调用；规划上由 resolve_data_config 在解析阶段统一完成，
        # 此处保留显式 import_target() 作为独立使用时的兜底
        target = import_target(self._target_)
        config_kwargs = {k: v for k, v in self.__dict__.items()
                        if k != "_target_" and not k.startswith("_")}
        return target(**config_kwargs, **kwargs)


@dataclass
class PackingConfig:
    """类型化 packing 配置基类。"""
    packed_sequence_size: int = 0
    packing_strategy: str = "thd"        # "thd" | "neat"
    max_packs: Optional[int] = None
    prepacked: bool = False

    def build(self, dataset, **kwargs):
        return dataset, None


@dataclass
class ThdPackingConfig(PackingConfig):
    """THD packing 配置。"""
    num_proc: int = 1

    def build(self, dataset, **kwargs):
        from hyper_models.components.datasets.llm.packed_sequence import pack_dataset
        from hyper_models.components.datasets.utils import packed_sequence_thd_collater
        ds = pack_dataset(
            dataset, **kwargs,
            packed_sequence_size=self.packed_sequence_size,
            max_packs=self.max_packs,
        )
        return ds, packed_sequence_thd_collater


@dataclass
class NeatPackingConfig(PackingConfig):
    """NEAT packing 配置。"""
    drop_long_samples: bool = True

    def build(self, dataset, **kwargs):
        from hyper_models.components.datasets.llm.neat_packing import neat_pack_dataset
        from hyper_models.components.datasets.utils import neat_packed_collater
        ds = neat_pack_dataset(
            dataset, **kwargs,
            pack_size=self.packed_sequence_size,
            max_packs=self.max_packs,
            drop_long_samples=self.drop_long_samples,
        )
        return ds, neat_packed_collater
```

---

## 19. Helper 函数签名

以下 helper 在前文被引用但未给出完整实现，此处统一声明签名。

```python
# ── 通用工具 ──

def split_into_chunks(tokens: list[int], max_length: int) -> list[list[int]]:
    """将 token 列表分块，每块不超过 max_length。"""
    ...


def add_flash_attention_kwargs_from_position_ids(batch: dict) -> dict:
    """根据 position_ids 为 flash attention 添加必要的 kwargs。"""
    ...


# ── _target_ 解析 ──

def _resolve_dataset_target(cfg) -> type | callable:
    """解析数据管道配置段的 `_target_` 为可调用对象。

    规划中的独立解析入口 resolve_data_config() 的内部环节：dataset 等数据管道
    配置段不在当前 TrainerConfig 字段内（resolve_root() 拒绝未知一级字段），
    不经 resolve_component()；机制复用 01 §2.4 的 import_target() + coerce_value()。
    """
    ...


# ── Dataset 元数据 ──

def _compute_dataset_sizes(paths: list[str]) -> list[int]:
    """从 Megatron .idx 文件读取每个数据集的实际 token 数。"""
    ...


# ── Tokenizer 构建 ──

def compute_trust_remote_code_from_model(cfg_model) -> bool:
    """根据 model config 推断是否需要 trust_remote_code=True。"""
    ...


def _get_model_name(cfg_model) -> str | None:
    """从 model config 提取 pretrained_model_name_or_path。"""
    ...


def _supports_seq_lens(model) -> bool:
    """判断 model.forward() 是否接受 seq_lens 参数。"""
    ...


# ── 分布式 / RNG 上下文 ──

class FirstRankPerNode:
    """上下文管理器：仅在该 node 的 local rank 0 上执行块内逻辑。"""
    ...


class ScopedRNG:
    """作用域 RNG：进入时按 seed + rank 派生独立随机状态，退出时恢复。"""
    ...


# ── PP causal mask ──

def _should_precompute_pp_causal_masks(model_config) -> bool:
    """判断 PP 模式下是否需要在 collate 阶段预计算 causal mask。"""
    ...


def add_causal_masks_to_batch(batch: dict, model_config) -> dict:
    """为 batch 中每个样本附加 causal mask（PP 第一 stage 预计算）。"""
    ...


# ── THD packing 内部 helper ──

def _fill_labels_with_cross_entropy_ignore_idx(labels, loss_mask):
    ...


def _should_stop_packing(max_packs, packs):
    ...


def _split_and_add_pack(current_pack, packs, previous_sample_boundary,
                         packed_sequence_size, padding_idx,
                         cross_entropy_ignore_idx, cp_size):
    ...


def _tensorize_and_pad_pack(current_pack, padding_idx,
                             packed_sequence_size, cross_entropy_ignore_idx, cp_size):
    ...


# ── packed_sequence_thd_collater 内部 helper ──

def get_pad_token_from_key(key, pad_token_ids):
    ...


def batchify(tensor):
    ...


def pad_within_micro(seq_lens_list, pad_value):
    ...


# ── VLM 内部 helper ──

def _conversation_has_media(conversation: list[dict]) -> bool:
    """判断 conversation 中是否包含媒体内容。"""
    ...


def inject_fake_image_into_conversation(conversation: list[dict]) -> list[dict]:
    """为纯文本 conversation 注入 fake image（FSDP 兼容性）。"""
    ...


# ── Chat Template 注册表 ──

# hyper_models/components/datasets/llm/chat_templates.py

CHAT_TEMPLATE_REGISTRY: dict[str, str] = {
    "default": "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}",
    "llama2": "[INST] {{ messages[0]['content'] }} [/INST] {{ messages[1]['content'] }}",
    "chatml": "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}",
}
```