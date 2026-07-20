# Hyper-Parallel 数据管道详细设计

> 参考实现：[AutoModel `recipes/llm/train_ft.py::build_dataloader()`](../../../auto_model/Automodel/nemo_automodel/recipes/llm/train_ft.py#L339)
> 上下文设计：[dual_mode_dtensor_parallel_strategy.md](../dual_mode_dtensor_parallel_strategy.md)

---

## 1. 模块职责

提供统一的数据管道，支持 **HF datasets** 和 **Megatron 二进制格式** 两种数据源，通过 `_target_` IoC 容器声明式配置。

### 核心文件

| 文件 | 职责 |
|------|------|
| `components/datasets/llm/dataloader.py` | `build_dataloader()` 统一入口（**架构决策**：放在 components 层而非 recipe 层，与 AutoModel 不同。AutoModel 的 `build_dataloader` 在 `recipes/llm/train_ft.py`。hyper_parallel 选择 components 层以提高跨 recipe 复用性） |
| `components/datasets/llm/packed_sequence.py` | THD packing |
| `components/datasets/llm/neat_packing.py` | NEAT packing (VLM) |
| `components/datasets/llm/megatron_dataset.py` | Megatron `.bin/.idx` 数据集封装 |
| `components/datasets/llm/megatron/sampler.py` | Megatron `MegatronPretrainingSampler` / `MegatronPretrainingRandomSampler` |
| `components/datasets/vlm/datasets.py` | VLM 数据集工厂函数 |
| `components/datasets/vlm/neat_packing_vlm.py` | VLM NEAT packing |
| `components/datasets/utils.py` | Collate 函数集合（`lm_collate`, `packed_sequence_thd_collater`, `neat_packed_collater`） |

### 涉及删除的旧代码

| 旧代码 | 替代方案 |
|--------|---------|
| `hyper_parallel/data/registry.py` — `DATASET_REGISTRY` 装饰器注册 | HF `datasets.load_dataset()` + `_target_` IoC |
| `hyper_parallel/data/dummy.py` | 封装为 `_target_: hyper_parallel.components.datasets.llm.DeterministicTokenDataset` |
| `hyper_parallel/data/hf.py` | 合并到 `build_dataloader()`，直接使用 `datasets.load_dataset()` |
| `hyper_parallel/data/preset_pt.py` | 封装为 `torch.utils.data.TensorDataset` 或独立路径 |
| `hyper_parallel/trainer/llm_trainer.py::_build_data_transform()` | 移到 `build_dataloader()` 内部 |
| `hyper_parallel/trainer/llm_trainer.py::_build_collate_fn()` | 移到 `components/datasets/utils.py` |

> **Megatron 数据源去留声明**（第六轮 N2 修复）：`hyper_parallel/data/megatron/` 下的
> `builder.py`（函数式 `build_megatron`）、`gpt_dataset.py`（`GPTDataset`）、
> `blendable_dataset.py`（`BlendableDataset`）、`indexed_dataset.py` **保留**，作为本文档新设计
> `MegatronPretraining`（§5.1）的底层实现——`MegatronPretraining.build()` 内部复用本地
> `GPTDataset`/`BlendableDataset`/`indexed_dataset` 完成二进制索引加载与样本切分，仅在外层
> 包装 `_target_` IoC 与 sampler 适配。即不删除本地 megatron 源码、也不 port AutoModel 全套，
> 而是让新 `MegatronPretraining` 复用本地实现。

---

### 命名空间与重命名声明

> **架构决策**：本文档所有代码路径统一使用 `hyper_parallel.components.datasets...`
> 命名空间（与真实 AutoModel 的 `nemo_automodel.components.datasets...` 对应）。
> 下文"与真实 `nemo_automodel/...` 对齐"的注释仅指对照真实源码核验签名，不代表
> 运行时 import `nemo_automodel`。
>
> **Tokenizer 重命名**：真实 AutoModel 的 `NeMoAutoTokenizer`（`nemo_automodel.
> _transformers.auto_tokenizer.NeMoAutoTokenizer`）在 hyper_parallel 中重命名为
> **`HyperAutoTokenizer`**（`hyper_parallel._transformers.auto_tokenizer.
> HyperAutoTokenizer`），二者等价。下文所有 `HyperAutoTokenizer.from_pretrained`
> 调用对应真实代码的 `NeMoAutoTokenizer.from_pretrained`。

---

## 2. 总入口调用时序：从 `recipe.setup()` 到 DataLoader 就绪

数据管道的全部构建工作在 `recipe.setup()` 中的 `build_dataloader()` 一次调用完成。以下是完整的调用树，数字序号表示执行顺序，缩进表示调用深度。

```
recipe.setup(cfg)                                                    # 03_training_loop.md
│
├─ ... (model, optimizer, loss, ... 等组件构建)
│
└─⑧/④.9 self.dataloader, self.tokenizer = build_dataloader(  # ⑧ = 02 编号, ④.9 = 01/03 canonical 编号              # 唯一入口
        cfg.dataset, cfg.dataloader, cfg.model, cfg.packed_sequence,
        seed, local_batch_size, global_batch_size,
        max_steps, val_check_interval, dp_rank, dp_world_size,
        pp_enabled, cp_size, model)
    │
    ├─⑧.1 kwargs, tokenizer = _build_tokenizer(cfg_model, cfg_ds)    # §4: 4 路分发
    │   ├─ 路径1: 无 tokenizer key → HyperAutoTokenizer.from_pretrained(model_name)
    │   ├─ 路径2: tokenizer 为 null → None
    │   ├─ 路径3: 有 tokenizer 无 _target_ → HyperAutoTokenizer.from_pretrained(**dict)
    │   └─ 路径4: 有 _target_ → cfg_ds.tokenizer.instantiate(trust_remote_code=...)
    │       → AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    │
    ├─⑧.2 ds = cfg_ds.instantiate(**kwargs)                          # §6: 数据集实例化
    │   │                                                            # _target_ 决定类型:
    │   ├─ _target_ == MegatronPretraining                            # §6.1: Megatron 路径
    │   │   ├─ global_batch_size / max_steps 等训练参数注入 kwargs
    │   │   ├─ cfg_ds.instantiate(**kwargs)
    │   │   │   → MegatronPretraining(paths=[...], seq_length=2048, ...)
    │   │   └─ ds.build()                                             # 构建 blend index
    │   │
    │   └─ _target_ == datasets.load_dataset (或其他)                  # HF 路径
    │       └─ with FirstRankPerNode():                                # 仅 rank0 触发下载
    │           cfg_ds.instantiate(**kwargs)
    │           → load_dataset(path="HuggingFaceFW/fineweb", split="train",
    │                          streaming=True, tokenizer=<tokenizer>)
    │
    ├─⑧.3 IterableDataset 分片                                        # §3.2 Step 3
    │   ├─ ds.shard(dp_world_size, dp_rank)          if callable
    │   └─ split_dataset_by_node(ds.dataset, ...)     if HF streaming
    │
    ├─⑧.4 Packed Sequence（可选）                                      # §7.1
    │   ├─ packing_strategy == "neat" → neat_pack_dataset(             # §8: VLM NEAT
    │   │       ds, split=cfg_ds.split, pack_size=packed_sequence_size,
    │   │       max_packs=..., padding_idx=..., drop_long_samples=...)
    │   │   configure_packing(attn_implementation=_attn_impl)
    │   │   cfg_dl.collate_fn = lambda b, _ai=_attn_impl: neat_packed_collater(b, attn_implementation=_ai)
    │   └─ else → pack_dataset(                                        # §7.1: THD
    │           ds, split=cfg_ds.split, packed_sequence_size=...,
    │           max_packs=..., padding_idx=..., cp_size=cp_size)
    │       → datasets.Dataset（input_ids/labels/position_ids/seq_lens/seq_lens_padded）
    │       cfg_dl.collate_fn 来自 YAML（packed_sequence_thd_collater），产出
    │           seq_lens/seq_lens_padded/qkv_format="thd"（§5.2）
    │
    ├─⑧.5 Sampler                                                      # §7.2
    │   ├─ MegatronPretraining → create_megatron_sampler(...)         # §6.2: MegatronPretrainingSampler
    │   │   dl_kwargs = {"batch_sampler": batch_sampler}
    │   ├─ map-style Dataset → StatefulDistributedSampler(...)        # §7.3: 断点续训兼容
    │   │   dl_kwargs = {"sampler": sampler, "batch_size": local_bs}
    │   └─ IterableDataset → dl_kwargs = {}                           # 无需 sampler
    │
    ├─⑧.6 Collate 函数                                                 # §5
    │   ├─ cfg_dl.collate_fn 有 _target_                               # §5: _target_ 模式
    │   │   → lambda batch: collate_cfg.instantiate(batch=batch)      # lazy per-batch
    │   ├─ cfg_dl.collate_fn 是 callable                               # 直接用
    │   ├─ 否则 → default_collater(tokenizer)                         # §5.1: 默认 padding
    │   └─ PP 模式 → AutoConfig.from_pretrained + chained_collate_fn  # §3.2 Step 6: 预计算 causal mask
    │       （base_collate → add_causal_masks_to_batch，用 hf_model_config 而非 model.config）
    │
    └─⑧.7 return cfg_dl.instantiate(**dl_kwargs), tokenizer           # 最终一步
        → StatefulDataLoader(dataset=ds, sampler=sampler, batch_size=1,
                             collate_fn=<fn>, num_workers=2, pin_memory=True)
```

**与 01 文档的时序衔接**：

```
main()
├─① load_yaml_config()           # 01 §2
├─② RecipeConfig(cfg)            # 01 §3
└─④ recipe.setup(cfg)            # 01 §4
    ├─④.4  model = ...           # 01 §4.1/§6
    ├─④.8  optimizer = ...       # 03_training_loop §9
    └─④.9  dataloader, tokenizer = build_dataloader(...)  ← 本文档入口
```

---

## 3. build_dataloader() 主流程

> **调用位置**: 时序树 ⑧ — `recipe.setup()` 中唯一入口，一次返回 `(DataLoader, tokenizer)`

### 3.1 函数签名

```python
# components/datasets/llm/dataloader.py

def build_dataloader(
    cfg_ds,                # Dataset ConfigNode（含 _target_）
    cfg_dl,                # DataLoader ConfigNode
    cfg_model,             # Model ConfigNode
    cfg_ps,                # PackedSequence ConfigNode
    seed: int,
    local_batch_size: int,
    global_batch_size: int,
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
    - datasets.load_dataset() → HF hub 数据集
    - MegatronPretraining → Megatron .bin/.idx 格式
    - 自定义 Dataset 类 → 任意 _target_ 可实例化的 Dataset

    返回: (DataLoader, tokenizer)
    """
```

### 3.2 完整实现（7 步流程）

```python
# 模块级 import（与真实 train_ft.py 对齐）：
# import logging, random, inspect
# from transformers import AutoConfig
# from transformers.tokenization_utils_base import PreTrainedTokenizerBase
# from torch.utils.data import DataLoader, IterableDataset
# from hyper_parallel.components.datasets.llm.megatron_dataset import MegatronPretraining
# from hyper_parallel.components.datasets.llm.megatron.sampler import create_megatron_sampler
# from hyper_parallel.components.datasets.llm.packed_sequence import pack_dataset
# from hyper_parallel.components.datasets.utils import (
#     default_collater, packed_sequence_thd_collater, neat_packed_collater,
#     add_causal_masks_to_batch,
# )
# from hyper_parallel.components.training.rng import ScopedRNG
# from hyper_parallel.components.distributed.utils import FirstRankPerNode
#   （属主文件 components/distributed/utils.py，由 06 实现；契约见 §10）
# from hyper_parallel.components.utils.model_utils import _supports_seq_lens
# from hyper_parallel.components.config.node import ConfigNode   # canonical 位置见 01
# from hyper_parallel.components.datasets.utils import (
#     _get_model_name,
#     compute_trust_remote_code_from_model,
#     _should_precompute_pp_causal_masks,
# )
# from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
#
# logger = logging.getLogger(__name__)

def build_dataloader(
    cfg_ds, cfg_dl, cfg_model, cfg_ps,
    seed, local_batch_size, global_batch_size,
    *,  # 以下 keyword-only：默认参数不得先于非默认参数（Python 语法要求）
    max_steps=None, val_check_interval=None,
    dp_rank, dp_world_size, pp_enabled, cp_size=1,
    model=None,
) -> tuple[DataLoader, PreTrainedTokenizerBase]:
    """构建 DataLoader。"""

    with ScopedRNG(seed=seed, ranked=True):
        # ── Step 1: 构建 Tokenizer ──
        kwargs, tokenizer = _build_tokenizer(cfg_model, cfg_ds)

        # ── Step 2: 实例化 Dataset ──
        if cfg_ds._target_ is MegatronPretraining:
            # Megatron 路径：传递 global_batch_size 等训练参数
            kwargs["global_batch_size"] = global_batch_size
            # 注入 micro_batch_size=local_batch_size：MegatronPretraining 默认 4，
            # 而 Step 5 的 create_megatron_sampler 用 local_batch_size 切 batch，
            # 二者必须一致，否则 sampler 语义与 dataset 记录的 micro_batch_size 漂移
            kwargs["micro_batch_size"] = local_batch_size
            kwargs["trainer_max_steps"] = max_steps
            kwargs["trainer_val_check_interval"] = val_check_interval
            ds = cfg_ds.instantiate(**kwargs)
            ds.build()  # Megatron 特有：构建 blend index
        else:
            # HF datasets 路径：每节点仅 rank 0 触发下载
            with FirstRankPerNode():
                ds = cfg_ds.instantiate(**kwargs)

        # ── Step 3: IterableDataset 分片 ──
        if isinstance(ds, IterableDataset):
            if callable(getattr(ds, "shard", None)):
                ds = ds.shard(dp_world_size, dp_rank)
            elif hasattr(ds, "dataset"):
                from datasets.distributed import split_dataset_by_node
                ds.dataset = split_dataset_by_node(
                    ds.dataset, world_size=dp_world_size, rank=dp_rank
                )

        # ── Step 4: Packed Sequence（可选） ──
        packed_sequence_size = getattr(cfg_ps, "packed_sequence_size", 0)
        packing_strategy = getattr(cfg_ps, "packing_strategy", "thd")
        prepacked_sequence = bool(getattr(cfg_ps, "prepacked", False))

        # 仅当模型可用且 forward 接受 seq_lens 时才启用 THD packing
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
                from hyper_parallel.components.datasets.llm.neat_packing import neat_pack_dataset
                from hyper_parallel.components.datasets.utils import neat_packed_collater
                # 跨层 import（数据层 → 模型层）：NEAT packing 需要从模型层获取
                # attention implementation 配置以决定 mask 格式（flash_attention_2
                # 保留 2D indexed mask；sdpa/eager 转 4D block-causal mask）。
                # 此耦合是 NEAT 算法自身的需求——packing 阶段的 mask 格式必须与
                # 模型 forward 使用的 attention 实现一致。
                from hyper_parallel.components.models.common.packing import (
                    configure_packing, get_attn_implementation,
                )
                # 注：models/common/packing.py 为新模块，01–06 其他文档未覆盖；
                # 其契约与最小实现要点见 §3.4。

                ds = neat_pack_dataset(
                    ds,
                    split=cfg_ds.split,
                    pack_size=packed_sequence_size,
                    max_packs=getattr(cfg_ps, "max_packs", None),
                    padding_idx=getattr(tokenizer, "pad_token_id", 0),
                    drop_long_samples=getattr(cfg_ps, "drop_long_samples", True),
                )
                _attn_impl = get_attn_implementation(cfg_model)
                configure_packing(attn_implementation=_attn_impl)
                # 绑定 attn_implementation，使 collater 产出正确的 mask 格式
                cfg_dl.collate_fn = lambda batch, _ai=_attn_impl: neat_packed_collater(
                    batch, attn_implementation=_ai
                )
            else:
                # "thd" —— collate_fn 由 YAML 声明（packed_sequence_thd_collater），
                # 产出 seq_lens/seq_lens_padded/qkv_format="thd"（§5.2），此处不在
                # 代码中覆写 cfg_dl.collate_fn
                ds = pack_dataset(
                    ds,
                    split=cfg_ds.split,
                    packed_sequence_size=packed_sequence_size,
                    max_packs=getattr(cfg_ps, "max_packs", None),
                    padding_idx=getattr(tokenizer, "pad_token_id", 0),
                    cp_size=cp_size,
                )

        # ── Step 5: Sampler ──
        if isinstance(ds, MegatronPretraining):
            # Megatron 路径：先取 split dataset 再取 len（build() 产出的是包装对象）
            # 用 .get() 取 splits_to_build：YAML 未配该 key 时 raise_on_missing_attr=True
            # 下直接属性访问会 AttributeError，且 None 兜底分支不可达
            split_to_get = cfg_ds.get("splits_to_build", None)
            if split_to_get is None:
                split_to_get = "train"
            elif isinstance(split_to_get, list):
                # splits_to_build 配成 list（如 [train, validation]）时，训练主
                # DataLoader 只取第一个 split；其余 split 由
                # build_validation_dataloader（§3.3）各自构建，不在此隐式展开
                split_to_get = split_to_get[0]
            ds = ds.get_dataset(split=split_to_get)
            dataloader_type = cfg_dl.get("dataloader_type", "single")
            if "dataloader_type" in cfg_dl:
                cfg_dl.__dict__.pop("dataloader_type", None)  # 避免与 batch_sampler 同传 ValueError
            batch_sampler = create_megatron_sampler(
                dataset_len=len(ds),
                micro_batch_size=local_batch_size,
                global_batch_size=global_batch_size,
                dataloader_type=dataloader_type,
                rank=dp_rank, world_size=dp_world_size,
            )
            dl_kwargs = {"batch_sampler": batch_sampler}
        elif not isinstance(ds, IterableDataset):
            # 清理与 batch_size/sampler 冲突的字段，避免 DataLoader ValueError
            shuffle = cfg_dl.get("shuffle", True)
            cfg_dl.__dict__.pop("shuffle", None)
            group_by_length = cfg_dl.get("group_by_length", False)
            cfg_dl.__dict__.pop("group_by_length", None)
            # drop_last 由 cfg_dl 驱动：训练默认 True；build_validation_dataloader
            # 会 replace(drop_last=False)，使验证集不丢尾部 batch（§3.3 契约）。
            # 此前此处硬编码 drop_last=True 会导致验证集尾部样本被 sampler 丢弃。
            drop_last = cfg_dl.get("drop_last", True)
            cfg_dl.__dict__.pop("drop_last", None)

            if group_by_length:
                from hyper_parallel.components.datasets.llm.length_grouped_sampler import (
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
            # sampler 与 DataLoader 两层 drop_last 保持一致（PP 强制 True）
            dl_kwargs = {
                "sampler": sampler,
                "batch_size": local_batch_size,
                "drop_last": drop_last or pp_enabled,
            }
        else:
            # IterableDataset：清理 shuffle 相关字段，不传 sampler
            cfg_dl.__dict__.pop("shuffle", None)
            cfg_dl.__dict__.pop("shuffle_buffer_size", None)
            shuffle = cfg_dl.get("shuffle", False)
            shuffle_buffer_size = cfg_dl.get("shuffle_buffer_size", 10000)
            if shuffle and hasattr(ds, "shuffle"):
                try:
                    ds = ds.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
                except Exception as e:
                    logger.warning(f"IterableDataset shuffle skipped: {e}")
            dl_kwargs = {}

        dl_kwargs["dataset"] = ds

        # ── Step 6: Collate ──
        # seq_divisor 用于 TP/CP 对齐 padding：使 seq_len 为 seq_divisor 的倍数
        seq_divisor = 2 * cp_size  # 至少对齐 CP；若上层可取 tp_size，则用 tp_size * cp_size

        # 处理 cfg_dl.collate_fn：_target_ ConfigNode → lazy instantiate；callable → 直接用
        if hasattr(cfg_dl, "collate_fn"):
            if getattr(cfg_dl.collate_fn, "_target_", None) is not None:
                collate_cfg = cfg_dl.collate_fn
                dl_kwargs["collate_fn"] = lambda batch: collate_cfg.instantiate(batch=batch)
            else:
                dl_kwargs["collate_fn"] = cfg_dl.collate_fn
            assert callable(dl_kwargs["collate_fn"]), "collate_fn must be callable"
        else:
            # tokenizer 为 None（路径 2）时 default_collater 内部以 pad_token_id=0
            # 兜底（见 §5.1 守卫），不会 None.pad_token_id AttributeError；
            # 但语义上无 tokenizer 的 padding 仅是兜底，建议 YAML 配置 tokenizer
            if tokenizer is None:
                logger.warning(
                    "No tokenizer configured; default_collater falls back to pad_token_id=0"
                )
            dl_kwargs["collate_fn"] = default_collater(tokenizer, seq_divisor=seq_divisor)

        # PP 模式：链式包装 collate 预计算 causal mask
        # 注意：用 AutoConfig.from_pretrained 而非 model.config，以避免 model 已被
        # parallelize 包装后 config 不可用；与真实 train_ft.py 一致
        #
        # 兼容性说明：PP + NEAT packing 目前未经充分测试，以下 PP causal mask
        # 预计算逻辑假设 batch 已通过标准 collate 处理并包含 input_ids/attention_mask；
        # NEAT collater 产出的 batch 结构（含 pixel_values/image_grids）与 PP
        # causal mask 注入的兼容性尚未验证。若需同时启用 PP 与 NEAT，建议先在
        # 单卡验证 collate 链的 batch 字段兼容性。
        if pp_enabled:
            from hyper_parallel.components.datasets.utils import add_causal_masks_to_batch

            try:
                hf_model_config = AutoConfig.from_pretrained(
                    _get_model_name(cfg_model),
                    trust_remote_code=compute_trust_remote_code_from_model(cfg_model),
                )
            except Exception:
                logger.warning("Failed to load model config for causal mask precomputation; skipping")
                hf_model_config = None

            if hf_model_config is not None and _should_precompute_pp_causal_masks(hf_model_config):
                # add_causal_masks_to_batch 期望 batch 已由 base_collate_fn 处理后
                # 包含以下字段：input_ids [B, S]、labels [B, S]、attention_mask [B, S]、
                # position_ids [B, S]（可选）。THD packing 路径下 batch 额外含
                # seq_lens/seq_lens_padded/qkv_format="thd"，这些字段不影响 causal
                # mask 注入（add_causal_masks_to_batch 仅依赖 input_ids 长度）。
                if "collate_fn" in dl_kwargs:
                    base_collate_fn = dl_kwargs["collate_fn"]

                    def chained_collate_fn(batch, base_fn=base_collate_fn, config=hf_model_config):
                        batch = base_fn(batch)
                        return add_causal_masks_to_batch(batch, model_config=config)

                    dl_kwargs["collate_fn"] = chained_collate_fn
                else:
                    dl_kwargs["collate_fn"] = lambda batch, config=hf_model_config: (
                        add_causal_masks_to_batch(batch, model_config=config)
                    )

        # ── Step 7: 构建 DataLoader ──
        dl_kwargs.update({
            "num_workers": cfg_dl.get("num_workers", 1),
            "pin_memory": cfg_dl.get("pin_memory", True),
        })
        # drop_last 已在 Megatron/map-style 分支按需设置；IterableDataset 不传 drop_last。
        # 注意：batch_sampler 与 drop_last 互斥（torch DataLoader ValueError），
        # Megatron 分支（batch_sampler）跳过此兜底，drop_last 由 create_megatron_sampler 内部处理
        if (
            "drop_last" not in dl_kwargs
            and "batch_sampler" not in dl_kwargs
            and cfg_dl.get("drop_last", True)
        ):
            dl_kwargs["drop_last"] = True
        return cfg_dl.instantiate(**dl_kwargs), tokenizer
```

### 3.3 Validation DataLoader

> **调用位置**: 03 `_run_validation_epoch` 前构建验证集 DataLoader。
> **canonical 归属**: `build_validation_dataloader` canonical 实现放在 02，03 调用。

`build_validation_dataloader` 复用 `build_dataloader` 主体，但参数化以下差异：
- `drop_last=False`（验证集不丢弃尾部 batch——该标记经 cfg_dl 传入
  `build_dataloader` Step 5，驱动 sampler 与 DataLoader 两层 drop_last，见 §3.2）
- `shuffle=False`（验证集顺序遍历）
- 不做 packing（`no_packing=True`，逐样本评估）
- 不创建 sampler 的断点续训状态（验证集无 checkpoint 恢复需求）

**调用约定（与 03 对齐，签名固定）**：前 7 个参数
`(cfg_ds, cfg_dl, cfg_model, cfg_ps, seed, local_batch_size, global_batch_size)`
按位置传，其余（`dp_rank / dp_world_size / pp_enabled / cp_size / model`）按关键字传；
不接收 `max_steps / val_check_interval`（内部以 None 传给 `build_dataloader`，
避免 `MegatronPretraining.build()` 误判训练步数调度）。返回
`dict[str, DataLoader]`，当前固定为 `{"validation": dl}`，03 按
`self.val_dataloaders = build_validation_dataloader(...)` 消费并遍历该 dict。

```python
# components/datasets/llm/dataloader.py

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

    通过覆盖 cfg_ps.packed_sequence_size=0 关闭 packing，再委托 build_dataloader。

    Returns:
        {"validation": DataLoader}
    """
    # 关闭 packing：复制一份避免污染训练用 config
    # ConfigNode.__init__ 收 dict（见 01 §2.3），且提供 replace(**overrides) 不可变更新
    # （等价于 to_dict()+ConfigNode(dict)）。此处用 replace 覆盖个别字段。
    from hyper_parallel.components.config.node import ConfigNode  # canonical 位置（01）

    cfg_ps_val = cfg_ps.replace(packed_sequence_size=0) if no_packing else cfg_ps

    # 覆盖 dataloader 的 drop_last / shuffle
    cfg_dl_val = cfg_dl
    if cfg_dl.get("drop_last", True) != drop_last or cfg_dl.get("shuffle", True) != shuffle:
        cfg_dl_val = cfg_dl.replace(drop_last=drop_last, shuffle=shuffle)

    dl, _ = build_dataloader(
        cfg_ds, cfg_dl_val, cfg_model, cfg_ps_val,
        seed, local_batch_size, global_batch_size,
        # max_steps=None：验证集不参与训练步数调度（真实 build_validation_dataloader
        # 传 cfg.get("step_scheduler.max_steps", None)，传 0 会让 MegatronPretraining.build()
        # 误判为 full-epoch 训练）
        max_steps=None, val_check_interval=None,
        dp_rank=dp_rank, dp_world_size=dp_world_size,
        pp_enabled=pp_enabled, cp_size=cp_size, model=model,
    )
    return {"validation": dl}
```

---

### 3.4 `components/models/common/packing.py` 模块契约（NEAT 分支依赖）

§3.2 NEAT 分支从 `hyper_parallel.components.models.common.packing` import
`configure_packing` / `get_attn_implementation`。该模块为新设模块（01–06 其他文档
未覆盖），此处补齐其契约与最小实现要点；实现归属模型层
（`components/models/common/`），数据层仅消费。

**存在理由**：NEAT packing 的 collater（`neat_packed_collater`）产出 attention
mask 的格式必须与模型 forward 实际使用的 attention 实现一致——
`flash_attention_2` 保留 2D indexed mask，`sdpa/eager` 需转为 4D block-causal
mask。数据层在构建期从 model config 读出实现名并写入一个进程级配置，collater
在每个 worker 进程内读取该配置决定 mask 格式。

```python
# components/models/common/packing.py

# 进程级 packing 配置（DataLoader worker fork/spawn 时随模块状态继承；
# spawn 模式下 collater 需能在 worker 内重新获取，故 configure_packing 须在
# DataLoader 构建前于主进程调用，且实现应保证幂等）
_PACKING_CONFIG: dict = {"attn_implementation": "sdpa"}


def get_attn_implementation(cfg_model) -> str:
    """从 model ConfigNode 推导 attention 实现名。

    契约：
    - 依次查 cfg_model.attn_implementation、cfg_model.config.attn_implementation
      （HF 风格嵌套 config），均未配置时返回 "sdpa" 作为安全默认。
    - 返回值 ∈ {"flash_attention_2", "sdpa", "eager"}（与 HF
      PretrainedConfig._attn_implementation 取值域一致）。
    """
    ...


def configure_packing(attn_implementation: str) -> None:
    """设置进程级 packing 配置，供 neat_packed_collater 在 collate 时读取。

    契约：
    - 幂等；重复调用以后一次为准。
    - 必须在构建 DataLoader 之前调用（worker 继承主进程模块状态）。
    - neat_packed_collater 内部读取本模块状态决定 mask 格式（§10
      neat_packed_collater 注释"必须在调用前先 configure_packing"即指此）。
    """
    _PACKING_CONFIG["attn_implementation"] = attn_implementation
```

---



> **调用位置**: 时序树 ⑧.1 — `build_dataloader()` Step 1

### 4.1 设计理念

Tokenizer 的类型和来源通过 YAML `_target_` 声明。AutoModel 有 4 条构建路径：

1. **无 tokenizer key** → 从 model 推断 → `HyperAutoTokenizer.from_pretrained(model_name)`
2. **tokenizer 为 null** → 跳过
3. **有 tokenizer 但无 `_target_`** → `HyperAutoTokenizer.from_pretrained(**tokenizer_dict)`
4. **有 `_target_`** → `cfg_ds.tokenizer.instantiate(trust_remote_code=...)`

```yaml
dataset:
  tokenizer:
    _target_: transformers.AutoTokenizer.from_pretrained
    pretrained_model_name_or_path: Qwen/Qwen3.5-4B
    trust_remote_code: true
```

#### 背景：为什么 AutoModel 要自己实现 Tokenizer

`NeMoAutoTokenizer`（hyper_parallel 中为 `HyperAutoTokenizer`）**不是"重新实现" tokenizer，而是在 HF `AutoTokenizer` 之上加了一个可扩展的分发层 + 训练框架所需的补丁层**。它存在的具体原因：

1. **按模型类型分发到专用 tokenizer（注册表机制）**

   真实代码 `auto_tokenizer.py:50-134` 中的分发逻辑是：

   - 先读 config 得到 `model_type`，查 `TokenizerRegistry` 里有没有注册的自定义实现（比如 Mistral 模型走 `tokenization_mistral_common.py` 的 `MistralCommonBackend`，用 `mistral-common` 官方分词，行为与 HF 的分词不同）；
   - 没有注册项才回退到 HF 的 tokenizer。

   HF 的 `AutoTokenizer` 不支持这种"按 `model_type` 插第三方后端"的扩展点，所以需要自己的门面。同时还保留了 `force_hf=True` 的逃生口，可以直接拿原始 HF tokenizer。

2. **训练侧强制 BOS/EOS 语义一致**

   默认实现 `NeMoAutoTokenizerWithBosEosEnforced` 的名字就说明了用途：有些 HF tokenizer（典型如 `GPT2Tokenizer`）不会自动加 BOS/EOS，而训练 pipeline（loss mask、序列拼接、`assistant_masks`）依赖边界 token 一定存在。它通过重写 `__call__` 和 `encode`（`nemo_auto_tokenizer.py:470-504`）保证 BOS/EOS 始终插入，并同步补齐 `attention_mask` / `assistant_masks`。

3. **集中消化 transformers v5 和各模型的兼容性坑**

   transformers 大版本升级（v5）以及各模型自带 tokenizer 的行为差异，会在训练数据侧引入大量边角问题。自研门面把这些兼容性 patch 集中在一个地方消化，而不是散落在各 dataset / collate 实现里。

> **设计推论**：hyper_parallel 保留 `HyperAutoTokenizer` 这一层，不是为了改分词行为，而是为了**继承上述扩展点**——路径 1/3 走它，路径 4 允许用户通过 `_target_` 显式绕过（如直接使用 `transformers.AutoTokenizer.from_pretrained`）。

### 4.2 实现：4 路分发

```python
def _build_tokenizer(cfg_model, cfg_ds) -> tuple[dict, PreTrainedTokenizerBase]:
    """从 ConfigNode 构建 tokenizer。

    与 AutoModel train_ft.py::_build_tokenizer 完全对齐的 4 路分发。
    """
    from hyper_parallel._transformers.auto_tokenizer import HyperAutoTokenizer

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
    # 注意：不能用 `"_target_" not in cfg_ds.tokenizer`——ConfigNode.__contains__
    # （01 §2.11）按 to_dict() 判定，而 to_dict() 排除 _target_，导致该条件恒真、
    # 路径 4 成为死代码。改用 getattr 显式探测 _target_ 属性。
    elif getattr(cfg_ds.tokenizer, "_target_", None) is None:
        tokenizer_dict = cfg_ds.tokenizer.to_dict()
        trust_remote_code = tokenizer_dict.pop("trust_remote_code", trust_remote_code)
        tokenizer = HyperAutoTokenizer.from_pretrained(
            **tokenizer_dict, trust_remote_code=trust_remote_code
        )
    # ── 路径 4: 有 _target_ → cfg_ds.tokenizer.instantiate(trust_remote_code=...) ──
    else:
        # 与真实 train_ft.py 一致：用 .instantiate() 走 _target_ 分发
        # （_target_ 可为 AutoTokenizer.from_pretrained 等），trust_remote_code
        # 从 tokenizer dict 弹出后作为 override 传入，避免重复 kwarg
        trust_remote_code = cfg_ds.tokenizer.to_dict().pop("trust_remote_code", trust_remote_code)
        tokenizer = cfg_ds.tokenizer.instantiate(trust_remote_code=trust_remote_code)

    # 设置 pad_token
    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset 构建时需要的 kwargs（根据 _target_ 签名决定是否注入 tokenizer）
    kwargs = {}
    if tokenizer is not None and cfg_ds._target_ is not None and callable(cfg_ds._target_):
        try:
            sig = inspect.signature(cfg_ds._target_)
            if "tokenizer" in sig.parameters:
                kwargs["tokenizer"] = tokenizer
        except (ValueError, TypeError):
            pass
    return kwargs, tokenizer
```

---

## 5. Collate 函数

> **调用位置**: 时序树 ⑧.6 — `build_dataloader()` Step 6

### 5.1 标准 LM Collate

> **命名约定**：对外名为 `default_collater`（`components/datasets/utils.py`），
> 与 01 §2.8 YAML `_target_: hyper_parallel.components.datasets.utils.default_collater`
> 及 §3.2 模块级 import 列表一致。曾用内部名 `_default_lm_collate` 已统一为
> `default_collater`，全文（含 §10 helper 清单）按此名引用。

```python
# components/datasets/utils.py

def default_collater(tokenizer, seq_divisor: int = 1):
    """对已有 labels 做 per-key padding（labels 由 dataset 提供，本函数不生成）。

    tokenizer 可为 None（tokenizer 构建路径 2：YAML 显式 tokenizer: null）——
    此时回退 pad_token_id=0，仅作为兜底；需要正确 padding 语义时应配置 tokenizer。
    """
    # None 守卫：避免 tokenizer=None 时 None.pad_token_id AttributeError
    pad_token_id = (getattr(tokenizer, "pad_token_id", None) or 0)

    # Per-key pad token IDs（与 AutoModel default_collater 对齐）
    ___PAD_TOKEN_IDS___ = {
        "input_ids": pad_token_id,
        "labels": -100,           # CrossEntropy ignore_index
        "attention_mask": 0,
        "position_ids": 0,
    }

    def collate(batch: list[dict]) -> dict:
        # 找到最长序列
        max_len = max(len(item["input_ids"]) for item in batch)

        # 确保 seq_len % seq_divisor == 0（TP/CP 要求）
        if seq_divisor > 1:
            remainder = max_len % seq_divisor
            if remainder:
                max_len += seq_divisor - remainder

        # position_ids 缺失时自动生成（不跳过）
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
                # 外层 `if key not in batch[0]: continue` 已保证 key 在 batch[0]；
                # 这里直接取 item[key]，非 tensor 先 as_tensor 以避免崩溃。
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

> **注**：PP causal mask 预计算不再使用独立的 `_make_pp_collate` helper。
> 真实 train_ft.py 在 `build_dataloader` Step 6 内联定义 `chained_collate_fn`
> （base_collate → `add_causal_masks_to_batch`），且 `model_config` 来自
> `AutoConfig.from_pretrained` 而非 `model.config`（见 §3.2 Step 6）。

### 5.2 Packed Sequence Collate（THD）

> **架构决策**：放弃自创的 `cu_seqlens` 契约，改用真实 AutoModel 的
> `seq_lens`/`seq_lens_padded`/`qkv_format="thd"` 契约（与
> `nemo_automodel.components.datasets.utils.packed_sequence_thd_collater` 完全对齐）。
> 05 `shard_batch_for_cp` 同步改为消费 `seq_lens`/`seq_lens_padded`。

```python
# components/datasets/utils.py

def packed_sequence_thd_collater(batch):
    """THD (Total, Hidden, Depth) packed sequence collater。

    与真实 AutoModel utils.py::packed_sequence_thd_collater 完全对齐：
    - 读取每项的 seq_lens / seq_lens_padded（由 pack_dataset 产出）
    - 对变长 seq_lens 用 -1000 sentinel pad 到 max_num_packs
    - 输出 qkv_format="thd"，不产出 cu_seqlens / _subseq_boundaries

    当 batch 项缺少 packed-sequence 元数据（如 ChatDataset 样本）时，collater
    合成缺失字段，使每个样本被视为单序列 "pack"——这使 THD 格式可用于 TE context
    parallelism 而无需 dataset 做实际 packing。

    Args:
        batch: list[dict]，预打包项含 input_ids/labels/position_ids/seq_lens/
            seq_lens_padded（由 pack_dataset 产出）；非打包项仅需 input_ids/labels。

    Returns:
        dict: input_ids/labels/position_ids [B, S]，
              seq_lens/seq_lens_padded [B, max_num_packs]（-1000 sentinel pad），
              qkv_format="thd"
    """
    pad_token_ids = None
    if len(batch) > 0 and "___PAD_TOKEN_IDS___" in batch[0]:
        pad_token_ids = batch[0].get("___PAD_TOKEN_IDS___")
        for item in batch:
            item.pop("___PAD_TOKEN_IDS___", None)

    if len(batch) == 0:
        return {}

    # 非打包数据（如 ChatDataset）：合成 seq_lens/seq_lens_padded/position_ids，
    # 使每个样本被视为单序列 pack
    if "seq_lens" not in batch[0]:
        input_ids_pad = get_pad_token_from_key("input_ids", pad_token_ids) or 0
        max_len = max(len(item["input_ids"]) for item in batch)

        for item in batch:
            cur_len = len(item["input_ids"])
            if "attention_mask" in item:
                actual_len = sum(item["attention_mask"])
                item.pop("attention_mask")
            else:
                actual_len = cur_len

            pad_amount = max_len - cur_len
            item["seq_lens"] = [actual_len]
            # seq_lens_padded 必须覆盖整个 padded 长度
            item["seq_lens_padded"] = [max_len]
            item["position_ids"] = list(range(max_len))

            if pad_amount > 0:
                item["input_ids"] = list(item["input_ids"]) + [input_ids_pad] * pad_amount
                item["labels"] = list(item["labels"]) + [-100] * pad_amount

    tokens = batchify(torch.stack([torch.tensor(x["input_ids"]) for x in batch]))
    labels = batchify(torch.stack([torch.tensor(x["labels"]) for x in batch]))
    position_ids = batchify(torch.stack([torch.tensor(x["position_ids"]) for x in batch]))

    # 变长 seq_lens 用 -1000 sentinel pad 到 max_num_packs
    seq_lens = batchify(torch.LongTensor(pad_within_micro([x["seq_lens"] for x in batch], -1000)))
    seq_lens_padded = batchify(torch.LongTensor(pad_within_micro([x["seq_lens_padded"] for x in batch], -1000)))

    return {
        "input_ids": tokens,
        "labels": labels,
        "position_ids": position_ids,
        "seq_lens": seq_lens,
        "seq_lens_padded": seq_lens_padded,
        "qkv_format": "thd",
    }
```

### 5.3 CP Batch Sharding (seq_lens 产出契约)

在 Context Parallel 场景下，每个 batch 需要在序列维度上切分给各个 CP rank。
`seq_lens`/`seq_lens_padded` 随 input_ids 一起切分，并在切分边界处调整
`seq_lens_padded`（CP padding 由 `pack_dataset` 在 packing 阶段已按 `2*cp_size`
对齐，见 §7.1）。

**02 仅声明 seq_lens 产出契约，`shard_batch_for_cp` 的 canonical 实现放在 05**
（`shard_batch_for_cp(batch, cp_mesh)`，按 CP chunk 切分 input_ids/labels/
position_ids 并同步切分 seq_lens/seq_lens_padded）。本文档不重复实现，避免与 05
的同名函数签名冲突。

#### seq_lens 产出契约

`packed_sequence_thd_collater` 产出的 batch 中 `seq_lens`/`seq_lens_padded`
字段需满足以下契约，供 05 `shard_batch_for_cp` 消费：

| 字段 | shape | dtype | 语义 |
|------|-------|-------|------|
| `seq_lens` | `[B, max_num_packs]`（-1000 sentinel pad） | int64 | 每个 pack 内各子序列的**原始**长度（不含 CP padding / pack padding） |
| `seq_lens_padded` | `[B, max_num_packs]`（-1000 sentinel pad） | int64 | 每个 pack 内各子序列的 **padded** 长度（含 CP padding；末项含 pack-level padding） |
| `qkv_format` | 标量 str | — | 恒为 `"thd"` |

| 属性 | 约定 |
|------|------|
| 来源 | `pack_dataset` 产出的 `datasets.Dataset`，每项含 `seq_lens`/`seq_lens_padded`（§7.1） |
| CP 对齐 | `pack_dataset` 已将每条子序列 pad 到 `2*cp_size` 的倍数（`seq_lens_padded` 反映此 padding） |
| sentinel | 变长 pack 数用 `-1000` pad 到 `max_num_packs`，下游需过滤 |

**CP 分片后形态**：05 `shard_batch_for_cp` 消费 `[B, max_num_packs]` 后，按 CP rank
的 chunk 区间切分 input_ids/labels/position_ids，并同步调整 seq_lens_padded
（落在 chunk 边界两侧的子序列按 chunk 内实际 token 数重算）。`seq_lens` 保持原始
子序列长度不变，仅按是否落入本 rank chunk 过滤。

```python
# 02 仅声明契约，不提供 shard_batch_for_cp 实现。
# canonical 实现见 05_dual_mode_dtensor_parallel_strategy.md §6.3.4：
#   def shard_batch_for_cp(batch, cp_mesh): ...
# 05 负责按 cp_mesh 切分 input_ids/labels/position_ids，并同步切分
# seq_lens/seq_lens_padded（[B, max_num_packs] → 各 rank 的子集）。
# 注：pack_dataset 已将每条样本打包到固定长度 S = packed_sequence_size，
# 并在 seq_lens_padded 中反映 CP padding，packed_sequence_thd_collater
# 直接 stack + sentinel pad 即可。
```

---

## 6. Megatron 数据集集成

> **调用位置**: 时序树 ⑧.2 Megatron 分支 — `cfg_ds._target_ == MegatronPretraining`

### 6.1 MegatronPretraining 封装

```python
# components/datasets/llm/megatron_dataset.py

class MegatronPretraining:
    """封装 Megatron 格式的预训练数据集。

    与真实 AutoModel nemo_automodel/components/datasets/llm/megatron_dataset.py
    ::MegatronPretraining 完全对齐的签名。Megatron 格式：
    - .bin 文件：预 tokenize 的 uint16/uint32 token 序列
    - .idx 文件：每个文档的 offset 索引

    YAML 配置示例：
        dataset:
          _target_: hyper_parallel.components.datasets.llm.MegatronPretraining
          paths:
            - "0.5 /path/to/dataset_a_text_document"
            - "0.5 /path/to/dataset_b_text_document"
          seq_length: 2048
          split: "900,50,50"
    """

    def __init__(
        self,
        paths: Path | list | dict[str, list],
        seq_length: int = 2048,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        create_attention_mask: bool = False,
        seed: int = 1234,
        split: str = "900,50,50",
        index_mapping_dir: Optional[str] = None,
        num_dataset_builder_threads: int = 1,
        num_train_samples: Optional[int] = None,
        num_val_samples: Optional[int] = None,
        num_test_samples: Optional[int] = None,
        trainer_max_steps: Optional[int] = None,
        trainer_val_check_interval: int = 1000,
        trainer_limit_val_batches: Union[int, float] = 1,
        trainer_limit_test_batches: Union[int, float] = 1,
        mmap_bin_files: bool = True,
        splits_to_build: Optional[Union[str, list[str]]] = None,
        object_storage_config: Optional[Union[dict, "ObjectStorageConfig"]] = None,
    ) -> None:
        """Pretraining dataset class for Megatron-LM datasets.

        Args:
            paths: 数据路径。可为单路径、路径列表（["w1", "path1", "w2", "path2"]
                或 ["path1", "path2"]）、dict-of-splits，或指向 JSON 文件的路径。
            seq_length: 序列长度。
            tokenizer: PreTrainedTokenizerBase 实例（数据已预 tokenize 时可省略）。
            micro_batch_size: 每 GPU batch size。
            global_batch_size: 全局 batch size。
            create_attention_mask: 是否生成 attention mask（fused/flash attention
                下不支持）。
            seed: GPT dataset 种子。
            split: "train,valid,test" 比例（逗号分隔三整数）；paths 为 dict 时忽略。
            index_mapping_dir: index mapping 文件写入目录。
            splits_to_build: 要构建的 split（"train"/"validation"/"test" 或其列表）；
                None 表示全部构建。Step 5 的
                get_dataset(split=cfg_ds.get("splits_to_build", None) 或 "train")
                依赖此字段。
            trainer_max_steps: 最大训练步数；None 或 -1 表示全 epoch。
            trainer_val_check_interval: 验证间隔。
        """
        # 编译 C++ helper（若未编译）
        if find_spec("hyper_parallel.components.datasets.llm.megatron.helpers_cpp") is None:
            compile_helper()

        # 归一化 object_storage_config
        if isinstance(object_storage_config, dict):
            object_storage_config = ObjectStorageConfig(**object_storage_config)

        if not isinstance(paths, (list, tuple, dict)):
            blend_config_or_none = try_load_blend_from_json(paths)
            paths = blend_config_or_none if blend_config_or_none is not None else get_list_of_files(paths)
        validate_dataset_asset_accessibility(paths, object_storage_config=object_storage_config)

        if isinstance(split, (list, tuple)):
            split = ", ".join(str(s) for s in split)

        build_kwargs = {}
        build_kwargs["mmap_bin_files"] = mmap_bin_files
        if isinstance(paths, dict):
            build_kwargs["blend_per_split"] = [
                get_blend_from_list(paths.get("train")),
                get_blend_from_list(paths.get("validation")),
                get_blend_from_list(paths.get("test")),
            ]
        else:
            paths, weights = get_blend_from_list(paths)
            if len(paths) == 1:
                weights = None
            build_kwargs["blend"] = [paths, weights]
            build_kwargs["split"] = split

        self.build_kwargs = build_kwargs
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.tokenizer = tokenizer
        self.create_attention_mask = create_attention_mask
        self.seed = seed
        self.split = split
        self.index_mapping_dir = index_mapping_dir
        self.num_dataset_builder_threads = num_dataset_builder_threads
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.splits_to_build = splits_to_build
        self.object_storage_config = object_storage_config
        self.trainer_max_steps = trainer_max_steps
        self.trainer_val_check_interval = trainer_val_check_interval
        self.trainer_limit_val_batches = trainer_limit_val_batches
        self.trainer_limit_test_batches = trainer_limit_test_batches

    def build(self):
        """构建 Megatron 数据集（委托 BlendedMegatronDatasetBuilder）。

        与真实 megatron_dataset.py::build() 一致：
        - blend 挂在 GPTDatasetConfig.blend（通过 gpt_dataset_config property），
          不进 builder 构造参数
        - builder 签名为 (sizes, is_built_on_rank, config, enabled_splits=None)，
          无 config_fn kwarg
        """
        train_iters = self.trainer_max_steps
        if train_iters is None or train_iters == -1:
            num_train_samples = None
        else:
            assert train_iters > 0
            num_train_samples = int(train_iters * self.global_batch_size)

        if self.num_train_samples is not None:
            num_train_samples = self.num_train_samples
            train_iters = int(num_train_samples / self.global_batch_size)

        if self.num_val_samples is not None:
            num_val_samples = self.num_val_samples
        elif train_iters is None or train_iters == -1:
            num_val_samples = None
        else:
            num_val_samples = (
                int(train_iters // self.trainer_val_check_interval)
                * self.trainer_limit_val_batches
                * self.global_batch_size
            )

        num_test_samples = self.num_test_samples if self.num_test_samples is not None else None
        train_valid_test_num_samples = [num_train_samples, num_val_samples, num_test_samples]

        self._train_ds, self._validation_ds, self._test_ds = BlendedMegatronDatasetBuilder(
            train_valid_test_num_samples,           # sizes: 每 split 的样本数
            is_built_on_rank=lambda: True,
            config=self.gpt_dataset_config,         # blend 通过 config.blend 传入
            enabled_splits=self.splits_to_build,
        ).build()

    def get_dataset(self, split: str):
        """返回指定 split 的数据集（split ∈ {"train","validation","test"}）。

        注意：此 `split` 参数表示数据集子集名称（"train"/"validation"/"test"），
        与 HF 路径中 `cfg_ds.split`（datasets 格式，如 "train[:1000]"）以及
        Megatron 配置中 `self.split`（ratio 字符串 "900,50,50"）语义不同。
        调用方 §3.2 Step 5 通过 `splits_to_build` 推导当前要使用的 split 名称，
        再传入本方法。三种 `split` 含义对照见 §6.1 `__init__` 参数文档。
        """
        mapping = {"train": "_train_ds", "validation": "_validation_ds", "test": "_test_ds"}
        assert split in ["train", "validation", "test"], f"Invalid split: {split}"
        if not hasattr(self, mapping[split]) or getattr(self, mapping[split]) is None:
            raise RuntimeError(
                f"Dataset for split {split} was not built. "
                f"Include '{split}' in splits_to_build to enable it."
            )
        return getattr(self, mapping[split])

    @property
    def gpt_dataset_config(self) -> "GPTDatasetConfig":
        """构造 GPTDatasetConfig（blend/reset_position_ids 等通过 build_kwargs 注入）。"""
        return GPTDatasetConfig(
            random_seed=self.seed,
            sequence_length=self.seq_length,
            tokenizer=self.tokenizer,
            path_to_cache=self.index_mapping_dir,
            reset_position_ids=False,
            create_attention_mask=self.create_attention_mask,
            reset_attention_mask=False,
            eod_mask_loss=False,
            num_dataset_builder_threads=self.num_dataset_builder_threads,
            object_storage_config=self.object_storage_config,
            **self.build_kwargs,
        )
```

### 6.2 Megatron Sampler

> **调用位置**: 时序树 ⑧.5 Megatron 分支

```python
# components/datasets/llm/megatron/sampler.py

def create_megatron_sampler(
    dataset_len: int,
    micro_batch_size: int,
    global_batch_size: int,
    dataloader_type: Literal["single", "cyclic"] = "single",
    drop_last: bool = True,
    pad_samples_to_global_batch_size: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> BaseMegatronSampler:
    """构建 Megatron 兼容 sampler。

    与真实 nemo_automodel/components/datasets/llm/megatron/sampler.py
    ::create_megatron_sampler 完全对齐。注意：不存在 BlendedMegatronSampler，
    真实实现用 MegatronPretrainingSampler / MegatronPretrainingRandomSampler。
    """
    if dataloader_type == "single":
        batch_sampler = MegatronPretrainingSampler(
            total_samples=dataset_len,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            data_parallel_rank=rank,
            data_parallel_size=world_size,
            drop_last=drop_last,
            pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
        )
    elif dataloader_type == "cyclic":
        batch_sampler = MegatronPretrainingRandomSampler(
            total_samples=dataset_len,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=rank,
            data_parallel_size=world_size,
            drop_last=drop_last,
        )
    else:
        raise Exception(f"{dataloader_type} dataloader type is not supported.")
    return batch_sampler
```

---

## 7. 序列并行三层粒度：CP、SP、THD

Context Parallel (CP)、Sequence Parallel (SP) 和 THD Packing 是三种
不同粒度的序列维度并行/组织策略，它们从粗到细逐层嵌套：

```
┌──────────────────────────────────────────────────────────────┐
│ 原始序列 [B, S]                                                │
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ Layer 1: CP (Context Parallel) — 粗粒度序列切分            ││
│  │   - 将 S 维度切分为 cp_size 个等长 chunk                   ││
│  │   - 每 rank 持有一个 contiguous chunk，attention 前        ││
│  │     all-gather K/V（05 D-01'' 定稿，flex_cp_allgather）     ││
│  │   - chunk 粒度: S / cp_size tokens                        ││
│  │   - seq_lens/seq_lens_padded 随 chunk 一起切分并重算       ││
│  └──────────────────────────────────────────────────────────┘│
│  ┌──────────────────────────────────────────────────────────┐│
│  │ Layer 2: SP (Sequence Parallel) — 中粒度边界内切分         ││
│  │   - 在 CP chunk 内部，沿序列维度做 TP-like 分片             ││
│  │   - LayerNorm/RMSNorm 沿序列维度分片以减少激活内存          ││
│  │   - 在 attention 前后做 all-gather / reduce-scatter       ││
│  └──────────────────────────────────────────────────────────┘│
│  ┌──────────────────────────────────────────────────────────┐│
│  │ Layer 3: THD (Total, Hidden, Depth) — 变长打包              ││
│  │   - 将多个不等长子序列打包进固定长度的 pack                 ││
│  │   - seq_lens/seq_lens_padded 标记子序列边界供 attention 使用  │
│  │   - 在 pack 内做 bin-packing，跨子序列无信息泄露            ││
│  └──────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

**三种粒度的关键区别**：

| 维度 | CP | SP | THD Packing |
|------|-----|-----|-------------|
| 粒度 | S / cp_size | 在 chunk 内按 tp 分片 | 变长子序列打包 |
| 通信模式 | all-gather K/V（collective，flex_cp_allgather，05 D-01'' 定稿） | all-gather + reduce-scatter (collective) | 无通信（纯数据组织） |
| 对模型影响 | attention 前 all-gather K/V 后按全序列计算（注入内部 attention） | LN/MLP 序列维度分片 | 无——对模型透明 |
| seq_lens/seq_lens_padded | 随 chunk 切分映射 | 不涉及 | 由 packing 阶段记录 |
| 典型配置 | cp_size=2~8 | 与 TP 共用 tp mesh | packed_sequence_size=8192~32768 |

### 7.1 THD Packing

```python
# components/datasets/llm/packed_sequence.py

from datasets import Dataset, DatasetDict  # HF datasets 类型

PACK_TYPE = dict  # pack 的内部结构：{"input_ids": [...], "labels": [...], "position_ids": [...], "seq_lens": [...]}
CROSS_ENTROPY_IGNORE_IDX = -100  # CrossEntropy loss 的 ignore_index

def pack_dataset(
    dataset,
    split,
    packed_sequence_size,
    max_packs=None,
    padding_idx=0,
    drop_long_samples=True,
    cp_size=1,
):
    """将多个短序列打包到一个 packed sequence 中（THD 格式）。

    与真实 AutoModel nemo_automodel/components/datasets/llm/packed_sequence.py
    ::pack_dataset 完全对齐。注意：真实实现不返回 PackedSequenceDataset 类，
    而是返回 datasets.Dataset（由 Dataset.from_dict 构造），每项含
    input_ids/labels/position_ids/seq_lens/seq_lens_padded。

    算法：
    1. 遍历 dataset，用 buffer 累积样本直到 packed_sequence_size
    2. 超长时按 drop_long_samples 决定丢弃或报错
    3. cp_size > 1 时，每条子序列 pad 到 2*cp_size 的倍数（写入 seq_lens_padded）
    4. 返回 datasets.Dataset，每项含 seq_lens（原始长度）/seq_lens_padded（padded 长度）

    Args:
        dataset: 实际数据集（可为 Dataset / DatasetDict）。
        split: 当 dataset 为 DatasetDict 时选择 split。
        packed_sequence_size: 每个 pack 的最大总 token 数。
        max_packs: 最大 pack 数（None 表示不限）。
        padding_idx: padding token id。
        drop_long_samples: 超长样本是否丢弃（False 则报错）。
        cp_size: Context Parallel size。>1 时每条子序列 pad 到 2*cp_size 倍数，
            写入 seq_lens_padded（而非强制 packed_sequence_size % cp_size == 0）。
    """
    packs: list[PACK_TYPE] = []
    if isinstance(dataset, DatasetDict):
        if split in dataset:
            dataset = dataset[split]
        else:
            logger.warning(f"Dataset {split} not found. Using entire dataset.")

    current_pack = {"input_ids": [], "labels": [], "position_ids": [], "seq_lens": []}
    previous_sample_boundary = 0
    cp_divisibility_factor = 2 * cp_size if cp_size > 1 else 1

    for sample in dataset:
        input_ids, labels = sample["input_ids"], sample["labels"]
        if loss_mask := sample.get("loss_mask", None):
            # 使用 .get() 而非 .pop() 以避免修改原始 dataset 的样本数据
            labels = _fill_labels_with_cross_entropy_ignore_idx(labels, loss_mask)

        seq_len = len(input_ids)
        if drop_long_samples and seq_len > packed_sequence_size:
            continue
        if seq_len > packed_sequence_size:
            raise ValueError(
                f"Dataset sample is too long ({seq_len} > {packed_sequence_size}). "
                "Please increase `packed_sequence_size`."
            )

        # CP padding：将子序列 pad 到 2*cp_size 倍数
        if cp_size > 1:
            cp_padded_len = ((seq_len + cp_divisibility_factor - 1) // cp_divisibility_factor) * cp_divisibility_factor
            cp_padding_amount = cp_padded_len - seq_len
            if cp_padding_amount > 0:
                input_ids = input_ids + [padding_idx] * cp_padding_amount
                labels = labels + [CROSS_ENTROPY_IGNORE_IDX] * cp_padding_amount

        current_pack["input_ids"] += input_ids
        current_pack["labels"] += labels
        # 注：position_ids 基于 len(input_ids) 生成，当 cp_size>1 时包括 CP padding
        # 部分的位置 ID。这些 padding 位置的 labels 为 CROSS_ENTROPY_IGNORE_IDX（-100），
        # 因此其 position_ids 语义偏差不会影响 loss 计算。严格做法应以原始 seq_len 生成
        # position_ids 再 append padding（position_ids=0），当前实现与真实 AutoModel 一致。
        current_pack["position_ids"] += [x % packed_sequence_size for x in range(len(input_ids))]
        current_pack["seq_lens"] += [seq_len]  # 始终存原始长度

        while len(current_pack["input_ids"]) > packed_sequence_size and not _should_stop_packing(max_packs, packs):
            current_pack = _split_and_add_pack(
                current_pack, packs=packs, previous_sample_boundary=previous_sample_boundary,
                packed_sequence_size=packed_sequence_size, padding_idx=padding_idx,
                cross_entropy_ignore_idx=CROSS_ENTROPY_IGNORE_IDX, cp_size=cp_size,
            )
        previous_sample_boundary = len(current_pack["input_ids"])
        if _should_stop_packing(max_packs, packs):
            break

    if len(current_pack["input_ids"]) > 0 and (max_packs is None or len(packs) < max_packs):
        packs.append(_tensorize_and_pad_pack(
            current_pack, padding_idx=padding_idx,
            packed_sequence_size=packed_sequence_size,
            cross_entropy_ignore_idx=CROSS_ENTROPY_IGNORE_IDX, cp_size=cp_size,
        ))

    if not packs:
        raise ValueError(
            f"No packs were produced: every sample was longer than "
            f"packed_sequence_size={packed_sequence_size} and was dropped, or the dataset was empty."
        )
    # 返回 datasets.Dataset（非 PackedSequenceDataset 类）
    return Dataset.from_dict({key: [pack[key] for pack in packs] for key in packs[0].keys()})
```

**`seq_lens`/`seq_lens_padded` 产出契约**：`pack_dataset` 返回的 `datasets.Dataset`
每个 item 含以下字段（由 `_tensorize_and_pad_pack` 写入，与真实实现一致）：

| 字段 | shape | 语义 |
|------|-------|------|
| `input_ids` | `[S]` (S = packed_sequence_size) | 打包后的 token 序列（含 padding） |
| `labels` | `[S]` | 对应 labels（padding 位为 -100） |
| `position_ids` | `[S]` | 每条子序列内从 0 重新计数的 position |
| `seq_lens` | `[num_packs]` | 每条子序列的**原始**长度（不含 CP padding / pack padding） |
| `seq_lens_padded` | `[num_packs]` | 每条子序列的 **padded** 长度（含 CP padding；末项含 pack-level padding） |

`packed_sequence_thd_collater`（§5.2）直接读取 `seq_lens`/`seq_lens_padded`，
用 -1000 sentinel pad 到 `max_num_packs` 后 stack。**不再产出 `cu_seqlens` 或
`_subseq_boundaries`**——这些是已废弃的自创契约。

### 7.2 Sampler

> **调用位置**: 时序树 ⑧.5 — `build_dataloader()` Step 5。根据 Dataset 类型选择：Megatron → MegatronPretrainingSampler、map-style → StatefulDistributedSampler（或 LengthGroupedSampler）、Iterable → 无

Megatron 分支的 sampler 实现见 §6.2 Megatron Sampler。

### 7.3 StatefulDistributedSampler

> **调用位置**: 时序树 ⑧.5 map-style 分支 — 断点续训兼容

```python
# 使用 torchdata 提供的断点续训兼容 sampler
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

# 支持 state_dict() / load_state_dict() 用于 checkpoint 恢复
sampler = StatefulDistributedSampler(
    dataset,
    seed=seed,
    drop_last=True,
    num_replicas=dp_world_size,
    rank=dp_rank,
    shuffle=True,
)
# → sampler.state_dict() → 保存到 checkpoint
# → sampler.load_state_dict(state) → 断点续训恢复位置
```

---

## 8. VLM 数据管道

> **调用位置**: 时序树 ⑧.2 / ⑧.4 NEAT 分支 — VLM 专用路径

### 8.1 统一 Conversation 格式

```python
# components/datasets/vlm/datasets.py

"""VLM 数据集工厂函数 —— 将各种 HF 数据集统一为 conversation 格式。"""

from datasets import Image as HfImage   # 延迟解码用（cast_column）

# 统一格式
# {
#     "conversation": [
#         {"role": "user", "content": [
#             {"type": "image", "image": PIL.Image},
#             {"type": "text", "text": "图片里有什么？"}
#         ]},
#         {"role": "assistant", "content": [
#             {"type": "text", "text": "图片里有一只猫。"}
#         ]}
#     ]
# }


def make_medpix_dataset(
    path_or_dataset="medpix-dataset/medpix-dataset",
    split="train",
):
    """MedPix 医疗影像数据集。"""
    from datasets import load_dataset
    from hyper_parallel.components.datasets.vlm._media import lazy_image

    dataset = load_dataset(path_or_dataset, split=split)
    # cast_column 将原始 bytes/image-path 列转为 HfImage(decode=False) 类型，
    # 这是 HF datasets 延迟解码的标准模式：数据在 __getitem__ 时才 decode 为 PIL.Image，
    # 避免一次性加载全部图片到内存。HfImage 即 datasets.Image（为避免与 PIL.Image
    # 冲突而显式别名）。
    dataset = dataset.cast_column("image_id", HfImage(decode=False))

    def transform(batch):
        return {"conversation": [
            [
                {"role": "user", "content": [
                    {"type": "image", "image": lazy_image(image)},
                    {"type": "text", "text": question}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": answer}
                ]}
            ]
            for image, question, answer in zip(
                batch["image_id"], batch["question"], batch["answer"]
            )
        ]}

    return dataset.with_transform(transform)  # 延迟在 __getitem__ 执行
```

### 8.2 PreTokenizedDatasetWrapper

```python
class PreTokenizedDatasetWrapper(torch.utils.data.Dataset):
    """在 DataLoader 的 __getitem__ 中做 tokenization。

    关键特性：
    - 超长样本：替换为随机样本重试（而非丢弃）
    - 纯文本样本：注入 fake image，避免 FSDP batch 结构不一致
    - labels：只对 assistant token 计算 loss
    """

    def __init__(self, dataset, processor, max_length, max_retries=10, truncate=False):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length
        self.max_retries = max_retries
        self.truncate = truncate

    def __getitem__(self, idx):
        from hyper_parallel.components.datasets.vlm.collate_fns import (
            _extract_media_from_conversations,
            build_labels_from_template,
        )
        from hyper_parallel.components.datasets.vlm.fake_image import (
            _conversation_has_media,
            inject_fake_image_into_conversation,
        )

        for attempt in range(self.max_retries):
            example = self.dataset[idx]
            example = _preload_media(example, self.processor)

            conversation = example["conversation"]

            # 纯文本样本注入 fake image
            if not _conversation_has_media(conversation):
                conversation = inject_fake_image_into_conversation(conversation)

            # apply_chat_template → processor → input_ids + pixel_values
            text = self.processor.apply_chat_template(
                [conversation], tokenize=False
            )
            # 媒体在 conversation content 内，而非 top-level example。
            # 用真实 _extract_media_from_conversations 提取（与真实
            # vlm/datasets.py::PreTokenizedDatasetWrapper.__getitem__ 一致）
            images, videos = _extract_media_from_conversations([conversation])
            result = self.processor(
                text=[text], images=images, videos=videos,
                return_tensors="pt",
            )

            seq_len = result["input_ids"].shape[1]
            if self.max_length and seq_len > self.max_length and not self.truncate:
                idx = random.randint(0, len(self.dataset) - 1)
                continue  # 重试

            # labels: 只对 assistant token 计算 loss
            labels = build_labels_from_template(
                result["input_ids"], [conversation], self.processor
            )
            result["labels"] = labels
            return result

    def __len__(self):
        return len(self.dataset)
```

---

## 9. 配置示例

### 9.1 HF datasets 配置

```yaml
recipe: FinetuneRecipe

dataset:
  _target_: datasets.load_dataset
  path: HuggingFaceFW/fineweb
  name: sample-10BT
  split: train
  streaming: true

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  batch_size: 1
  num_workers: 2
  pin_memory: true

packed_sequence:
  packed_sequence_size: 8192
  packing_strategy: thd
```

### 9.2 Megatron 数据集配置

> **验证清单**：YAML 中 `_target_: hyper_parallel.components.datasets.llm.MegatronPretraining`
> 使用短路路径（不含 `.megatron_dataset` 模块名），要求
> `hyper_parallel/components/datasets/llm/__init__.py` 从 `megatron_dataset`
> 模块重导出 `MegatronPretraining` 类。若 `__init__.py` 仅含空文件或未显式导出
> 该类，ConfigNode 的 `instantiate()` 将因 `_target_` 解析失败而报错。
> 构建前请确认该 `__init__.py` 包含 `from hyper_parallel.components.datasets.llm.megatron_dataset import MegatronPretraining`。

```yaml
recipe: FinetuneRecipe

dataset:
  _target_: hyper_parallel.components.datasets.llm.MegatronPretraining
  paths:
    - "0.5 /data/corpus_a_text_document"
    - "0.5 /data/corpus_b_text_document"
  seq_length: 2048
  split: "900,50,50"
  splits_to_build: train

dataloader:
  _target_: torch.utils.data.DataLoader
  num_workers: 2
```

### 9.3 VLM 数据集配置

```yaml
recipe: FinetuneRecipe

dataset:
  _target_: hyper_parallel.components.datasets.vlm.make_medpix_dataset
  path_or_dataset: mmoukouba/MedPix-VQA
  split: train

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  batch_size: 1
  num_workers: 1
```

---

## 10. Helper 函数签名

以下 helper 在前文被引用但未给出完整实现，此处统一声明签名（body 以 `...` 表示，
具体实现归属见各函数注释）。列出签名是为了让调用点的 arity 与类型可据文档核对。

```python
# ── Collate 函数归属清单 ──
# 以下 collater 均属主 hyper_parallel/components/datasets/utils.py：
#   default_collater            —— §5.1（标准 LM padding collater；对外名，
#                                  01 §2.8 YAML 与 §3.2 import 列表均用此名）
#   packed_sequence_thd_collater —— §5.2（THD packing，完整实现见 §5.2）
#   neat_packed_collater        —— 下方签名（VLM NEAT）
#   add_causal_masks_to_batch   —— 下方签名（PP causal mask 预计算）


# ── ConfigNode 解析 ──
# 注：通用 _target_ 字符串 → callable 的解析由 01 §2.4 的 _resolve_target(dotted_path: str)
# 统一权威实现。本函数仅是 dataset 侧的薄封装：当 ConfigNode 已缓存解析后的
# callable 时直接返回，否则将字符串交给 _resolve_target 解析。避免在 02 重复
# 定义一份 import-path 解析逻辑导致两套行为漂移。
from hyper_parallel.components.config.node import _resolve_target  # 01 §2.4 canonical

def _resolve_dataset_target(cfg) -> type | callable:
    """解析 ConfigNode 的 _target_ 为可调用对象。

    优先返回 cfg._target_（ConfigNode 在 _wrap 阶段已将其解析为类/可调用对象，
    见 01 §2.3）；若因字符串配置等原因仍未解析，则按字符串 import path 交给
    _resolve_target 解析（如 "datasets.load_dataset"）。
    用于避免 `cfg_ds._target_ == MegatronPretraining` 在字符串配置下的误判。
    """
    target = getattr(cfg, "_target_", None)
    if callable(target):
        return target
    if isinstance(target, str):
        return _resolve_target(target)
    raise ValueError(f"_resolve_dataset_target: cannot resolve _target_={target!r}")


# ── Dataset 元数据 ──
def _compute_dataset_sizes(paths: list[str]) -> list[int]:
    """从 Megatron .idx 文件读取每个数据集的实际 token 数（blend_sizes）。"""
    ...


# 注：_build_cu_seqlens_from_boundaries 已删除——自创 cu_seqlens 契约被废弃，
# 改用真实 seq_lens/seq_lens_padded 契约（§5.2）。


# ── Tokenizer 构建（§4.2 引用） ──
def compute_trust_remote_code_from_model(cfg_model) -> bool:
    """根据 model config 推断是否需要 trust_remote_code=True。

    与真实 train_ft.py::compute_trust_remote_code_from_model 一致：
    优先读 cfg_model.trust_remote_code，其次 cfg_model.config.trust_remote_code，
    最后 resolve_trust_remote_code(_get_model_name(cfg_model))。
    """
    ...


def _get_model_name(cfg_model) -> str | None:
    """从 model config 提取 pretrained_model_name_or_path（或等价字段）。"""
    ...


def _supports_seq_lens(model) -> bool:
    """判断 model.forward() 是否接受 seq_lens 参数（决定是否启用 THD packing）。

    与真实 nemo_automodel/components/utils/model_utils.py::_supports_seq_lens 一致。
    """
    ...


# ── 分布式 / RNG 上下文（§3.2 引用） ──
# 属主文件：hyper_parallel/components/distributed/utils.py
# （该文件目前不存在，06 实现分布式基础设施时需补建；02 的 import 路径
#  `from hyper_parallel.components.distributed.utils import FirstRankPerNode`
#  保持不变）
class FirstRankPerNode:
    """上下文管理器：仅在该 node 的 local rank 0 上执行块内逻辑（用于 HF 下载）。

    契约：进入时 local_rank != 0 的进程阻塞（barrier 或条件等待），local_rank == 0
    的进程执行块内逻辑（如触发 HF datasets 下载/缓存）；退出时所有进程同步一次
    （barrier），保证非 0 号进程进入后续逻辑时缓存已就绪。多节点下每节点各有一个
    "first rank"（按 node 内 local rank 判定，而非全局 rank 0）。
    """
    def __enter__(self): ...
    def __exit__(self, *exc): ...


class ScopedRNG:
    """作用域 RNG：进入时按 seed + rank 派生独立随机状态，退出时恢复。"""
    def __init__(self, seed: int, ranked: bool = False): ...
    def __enter__(self): ...
    def __exit__(self, *exc): ...


# ── PP causal mask（§3.2 / §5.1 引用） ──
def _should_precompute_pp_causal_masks(model_config) -> bool:
    """判断 PP 模式下是否需要在 collate 阶段预计算 causal mask。

    与真实 train_ft.py::_should_precompute_pp_causal_masks 一致：
    `return getattr(model_config, "model_type", None) != "deepseek_v4"`
    注意：model_config 应来自 AutoConfig.from_pretrained，而非已 parallelize
    的 model.config（§3.2 Step 6 已用 AutoConfig.from_pretrained）。
    """
    ...


def add_causal_masks_to_batch(batch: dict, model_config) -> dict:
    """为 batch 中每个样本附加 causal mask（PP 第一 stage 预计算）。

    与真实 nemo_automodel/components/datasets/utils.py::add_causal_masks_to_batch 一致。
    """
    ...


# ── NEAT packing（§3.2 / 时序树 ⑧.4 引用） ──
def neat_pack_dataset(
    dataset,
    split: str,
    pack_size: int,
    max_packs: int | None = None,
    padding_idx: int = 0,
    drop_long_samples: bool = False,
):
    """VLM NEAT packing：将变长 conversation 打包到固定 pack_size。

    与真实 nemo_automodel/components/datasets/llm/neat_packing.py
    ::neat_pack_dataset 签名完全对齐（含 split / drop_long_samples）。
    返回 datasets.Dataset，每项含 input_ids/labels/attention_mask/position_ids。
    """
    ...


def neat_packed_collater(batch: list[dict], attn_implementation: str = "sdpa") -> dict:
    """NEAT packed sequence collater（VLM 专用，处理 image grid / pixel_values）。

    与真实 nemo_automodel/components/datasets/utils.py::neat_packed_collater 一致：
    attn_implementation 决定 mask 格式（flash_attention_2 保留 2D indexed mask；
    sdpa/eager 转 4D block-causal mask）。必须在调用前先 configure_packing。
    """
    ...


# ── THD packing 内部 helper（§7.1 引用） ──
def _fill_labels_with_cross_entropy_ignore_idx(
    labels: list[int], loss_mask: list[int],
) -> list[int]:
    """根据 loss_mask 将不参与 loss 的位置设为 CROSS_ENTROPY_IGNORE_IDX（-100）。
    loss_mask[i] == 0 的 labels 位置替换为 -100。
    """
    ...


def _should_stop_packing(max_packs: int | None, packs: list) -> bool:
    """判断是否已达到最大 pack 数上限。"""
    ...


def _split_and_add_pack(
    current_pack: dict, packs: list, previous_sample_boundary: int,
    packed_sequence_size: int, padding_idx: int,
    cross_entropy_ignore_idx: int, cp_size: int,
) -> dict:
    """将当前累积 buffer 按 packed_sequence_size 切分并产出新 pack。
    返回切分后剩余的 current_pack（超过 packed_sequence_size 的部分）。
    """
    ...


def _tensorize_and_pad_pack(
    current_pack: dict, padding_idx: int,
    packed_sequence_size: int, cross_entropy_ignore_idx: int, cp_size: int,
) -> dict:
    """将 current_pack 转为 tensor 并 pad 到 packed_sequence_size，
    同时写出 seq_lens_padded（包含 CP padding 后的各子序列长度）。
    """
    ...


# ── packed_sequence_thd_collater 内部 helper（§5.2 引用） ──
def get_pad_token_from_key(key: str, pad_token_ids: dict | None) -> int:
    """从 ___PAD_TOKEN_IDS___ dict 获取指定 key 的 pad token id。"""
    ...


def batchify(tensor: torch.Tensor) -> torch.Tensor:
    """将输入 tensor 确保为 batch-first 布局（2D → 保持，3D+ → 展平前两维）。
    与真实 nemo_automodel/components/datasets/utils.py::batchify 一致。
    """
    ...


def pad_within_micro(
    seq_lens_list: list[list[int]], pad_value: int,
) -> list[list[int]]:
    """将一个 micro-batch 内变长的 seq_lens 列表 pad 到相同长度，
    使用 pad_value 填充（-1000 sentinel）。
    """
    ...


# ── VLM 内部 helper（§8.2 引用） ──
def _preload_media(example: dict, processor) -> dict:
    """将 example conversation 中的媒体（PIL.Image / bytes）预加载并解码，
    供后续 processor（）调用。对于延迟解码（HfImage(decode=False)）的数据集，
    此步骤负责触发实际解码。
    """
    ...


# ── pack_dataset（§7.1 已给出完整实现） ──
# pack_dataset(dataset, split, packed_sequence_size, max_packs=None,
#              padding_idx=0, drop_long_samples=True, cp_size=1) -> datasets.Dataset
```
