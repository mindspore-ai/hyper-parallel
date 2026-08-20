# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""DiTTrainer — Diffusion Transformer training for Qwen-Image.

Composition pattern: holds a BaseTrainer and overrides data pipeline steps.
"""

import glob
import logging
import os
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from hyper_parallel.trainer.base import BaseTrainer

logger = logging.getLogger(__name__)


class DiTTrainer:
    """Trainer for Qwen-Image DiT diffusion training.

    Supports:
    - ``data.type = "dummy_dit"``: deterministic synthetic diffusion tensors
      for quick single-card validation and loss alignment.

    Composition pattern — delegates training loop to BaseTrainer.
    """

    def __init__(self, args):
        self.base = BaseTrainer(args)

        # 13-step init — call base methods, override data steps
        self.base._setup()
        self.base._build_model()
        # 注意：不要手动 .to(device)，_build_parallelized_model 会处理 meta → npu
        self.base._freeze_model()
        self._build_model_assets()
        self._build_data_transform()
        self._build_dataset()
        self._build_collate_fn()
        self._build_dataloader()
        self.base._build_parallelized_model()
        self._build_optimizer()
        self.base._build_lr_scheduler()
        self.base._build_training_context()
        self.base._init_callbacks()
        self.base.on_init_end()

    # ------------------------------------------------------------------
    # Overridden _build_* methods
    # ------------------------------------------------------------------

    def _build_model_assets(self):
        """DiT does not need tokenizer or processor for dummy data."""
        self.base.tokenizer = None
        self.base.processor = None

    def _build_data_transform(self):
        """No offline data transform for dummy tensors."""
        self.base.data_transform = None

    def _build_dataset(self):
        """Build deterministic dummy DiT dataset with real diffusion targets.

        Pre-generates a fixed data pool using a fixed seed so that
        single-card and multi-card training see identical samples.
        """
        data_type = getattr(self.base.args.data, "type", "dummy_dit")

        model_cfg = self.base.args.model
        cond_dim = getattr(model_cfg, "joint_attention_dim", 3584)
        seq_len = 77
        base_seed = int(getattr(self.base.args, "seed", 42))
        max_steps = getattr(self.base.args.train, "max_steps", 100)

        if data_type == "coco_parquet":
            self._build_parquet_dataset(max_steps)
        elif data_type == "coco_dit":
            self._build_coco_dataset(cond_dim, seq_len, base_seed, max_steps)
        elif data_type == "dummy_dit":
            self._build_dummy_dataset(model_cfg, cond_dim, seq_len, base_seed, max_steps)
        else:
            raise NotImplementedError(
                f"DiTTrainer supports 'dummy_dit'/'coco_dit'/'coco_parquet', got '{data_type}'"
            )

    def _build_dummy_dataset(self, model_cfg, cond_dim, seq_len, base_seed, max_steps):
        """Deterministic dummy DiT dataset with flow-matching targets."""
        in_ch = getattr(model_cfg, "in_channels", 64)
        out_ch = getattr(model_cfg, "out_channels", 64)
        height = getattr(model_cfg, "height", 256)
        width = getattr(model_cfg, "width", 256)
        latent_h = height // 8
        latent_w = width // 8

        g = torch.Generator().manual_seed(base_seed)
        samples = []
        for _ in range(100):
            clean = torch.randn(in_ch, latent_h, latent_w, generator=g)
            eps = torch.randn(out_ch, latent_h, latent_w, generator=g)
            condition = torch.randn(seq_len, cond_dim, generator=g)
            ts = torch.randint(1, 1000, (1,), generator=g).squeeze(0)
            t_norm = ts.float() / 1000.0
            x_t = (1.0 - t_norm) * clean + t_norm * eps
            velocity = eps - clean
            samples.append({
                "latent": x_t, "timestep": ts,
                "condition": condition, "target_noise": velocity,
                "labels": torch.tensor(1, dtype=torch.long),
            })

        samples = samples * 2

        class DummyDiTDataset(Dataset):
            def __init__(self, samples):
                self.samples = samples
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, idx):
                return self.samples[idx % len(self.samples)]

        self.base.train_dataset = DummyDiTDataset(samples)
        self.base.state.max_steps = max_steps
        logger.info_rank0(
            f"DiT dummy dataset: {len(samples)} samples, "
            f"latent=({in_ch},{latent_h},{latent_w}) cond=({seq_len},{cond_dim})"
        )

    def _build_coco_dataset(self, cond_dim, seq_len, base_seed, max_steps):
        """Real COCO images (packed VAE latents) + dummy text embeddings."""
        cache_path = getattr(self.base.args.data, "train_path", None)
        if not cache_path:
            raise ValueError("data.train_path must be set for data.type=coco_dit")
        pth_files = sorted(glob.glob(os.path.join(cache_path, "*.pth")))
        if not pth_files:
            raise FileNotFoundError(f"No .pth files found in {cache_path}")

        class CocoDiTDataset(Dataset):
            """Map-style dataset over packed-VAE .pth files with on-the-fly
            flow-matching noise targets."""
            def __init__(self, files, cond_dim, seq_len, seed):
                self.files = files
                self.cond_dim = cond_dim
                self.seq_len = seq_len
                self.seed = seed

            def __len__(self):
                return len(self.files)

            def __getitem__(self, idx):
                data = torch.load(self.files[idx])
                clean = data["latent_clean"]
                g = torch.Generator().manual_seed(self.seed + idx)
                eps = torch.randn(*clean.shape, generator=g)
                ts = torch.randint(1, 1000, (1,), generator=g).squeeze(0)
                t_norm = ts.float() / 1000.0
                x_t = (1.0 - t_norm) * clean + t_norm * eps
                velocity = eps - clean
                if "text_embed" in data and data["text_embed"] is not None:
                    condition = data["text_embed"]
                else:
                    condition = torch.randn(self.seq_len, self.cond_dim, generator=g)
                return {
                    "latent": x_t, "timestep": ts,
                    "condition": condition, "target_noise": velocity,
                    "labels": torch.tensor(1, dtype=torch.long),
                }

        self.base.train_dataset = CocoDiTDataset(pth_files, cond_dim, seq_len, base_seed)
        self.base.state.max_steps = max_steps
        logger.info_rank0(
            f"COCO dataset: {len(pth_files)} samples from {cache_path}, "
            f"cond=({seq_len},{cond_dim})"
        )

    def _build_parquet_dataset(self, max_steps):
        """Read pre-generated data from parquet via HuggingFace Datasets.

        Both HP and VeOmni load the same parquet through HuggingFace Datasets,
        ensuring byte-identical training inputs for cross-framework alignment.
        """
        try:
            from datasets import load_dataset  # pylint: disable=import-outside-toplevel
            import pickle as pk  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError("datasets package required: pip install datasets") from exc

        parquet_path = getattr(self.base.args.data, "train_path", None)
        if not parquet_path:
            raise ValueError("data.train_path must point to a .parquet file for data.type=coco_parquet")

        hf_ds = load_dataset("parquet", data_files=parquet_path, split="train")

        class ParquetDataset(Dataset):
            """Map-style dataset reading pre-generated rows from a parquet file."""
            def __init__(self, hf_ds):
                self.hf_ds = hf_ds
            def __len__(self):
                return len(self.hf_ds)
            def __getitem__(self, idx):
                row = self.hf_ds[idx]
                hidden = pk.loads(row["hidden_states"]).squeeze(0)
                target = pk.loads(row["training_target"]).squeeze(0)
                c, h, w = 64, 16, 16
                return {
                    "latent": hidden.reshape(h, w, c).permute(2, 0, 1).float(),
                    "timestep": pk.loads(row["timestep"]).squeeze(0),
                    "condition": pk.loads(row["encoder_hidden_states"]).squeeze(0).float(),
                    "target_noise": target.reshape(h, w, c).permute(2, 0, 1).float(),
                    "labels": torch.tensor(1, dtype=torch.long),
                }

        self.base.train_dataset = ParquetDataset(hf_ds)
        self.base.state.max_steps = max_steps
        logger.info_rank0(
            f"Parquet dataset: {len(hf_ds)} samples from {parquet_path}"
        )

    def _build_collate_fn(self):
        """Stack fixed-size tensors (no padding needed for dummy data)."""

        def _dit_collate(batch):
            return {
                "latent": torch.stack([x["latent"] for x in batch]),
                "timestep": torch.stack([x["timestep"] for x in batch]),
                "condition": torch.stack([x["condition"] for x in batch]),
                "target_noise": torch.stack([x["target_noise"] for x in batch]),
                "labels": torch.stack([x["labels"] for x in batch]),
            }

        self.base.collate_fn = _dit_collate

    def _build_dataloader(self):
        """Build DataLoader with no distributed sharding.

        All ranks (single-card or multi-card) iterate over the *same*
        samples in the *same* order, which is required for step-by-step
        loss alignment between single-card and FSDP training.

        Sampler uses ``shuffle=True`` with ``seed`` from the config so that
        the index sequence matches VeOmni's
        ``StatefulDistributedSampler`` (which inherits
        ``torch.utils.data.distributed.DistributedSampler``); both produce
        the same ``torch.randperm(len(dataset), generator=g.manual_seed(seed))``
        sequence, ensuring per-step timestep/noise inputs are byte-identical
        for cross-framework loss alignment.
        """
        # 先调用框架默认方法（正确设置 _grad_accum 等属性）
        self.base._build_dataloader()  # pylint: disable=protected-access

        # 然后替换 sampler 为不分片的
        old_dl = self.base.train_dataloader
        dataset = old_dl.dataset
        batch_size = old_dl.batch_size

        sampler = DistributedSampler(
            dataset,
            num_replicas=1,   # 关键：不分片，所有 rank 按同样顺序取
            rank=0,
            shuffle=True,
            seed=int(getattr(self.base.args.train, "seed", 42)),
        )

        self.base.train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=old_dl.collate_fn,
            drop_last=old_dl.drop_last,
        )

    def _build_optimizer(self):
        """Override base._build_optimizer to use fused AdamW (matches VeOmni).

        Replicates BaseTrainer._build_optimizer but constructs ``torch.optim.AdamW``
        with ``fused=True, foreach=False`` to match VeOmni's
        ``build_optimizer(fused=True)`` path, ensuring identical optimizer
        numerics across frameworks for loss alignment.
        """
        lr = getattr(self.base.args.train.optimizer, 'lr', 1e-4)
        weight_decay = getattr(self.base.args.train.optimizer, 'weight_decay', 0.01)

        decay_keywords = ("bias", "layernorm", "norm", "rmsnorm")

        def _is_no_decay(name: str) -> bool:
            lname = name.lower()
            return any(kw in lname for kw in decay_keywords)

        decay_params = []
        no_decay_params = []
        seen_ids = set()
        for n, p in self.base.model.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in seen_ids:
                continue
            seen_ids.add(id(p))
            if _is_no_decay(n):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        adam_eps = getattr(self.base.args.train.optimizer, 'eps', 1e-8)
        adam_betas = getattr(self.base.args.train.optimizer, 'betas', (0.9, 0.999))
        self.base.optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=adam_betas,
            eps=adam_eps,
            foreach=False,
            fused=True,
        )
        logger.info(
            "Optimizer (DiT override): AdamW fused=True lr=%.2e wd=%.3g "
            "decay_params=%d no_decay_params=%d",
            lr, weight_decay, len(decay_params), len(no_decay_params),
        )

    # ------------------------------------------------------------------
    # Delegated methods
    # ------------------------------------------------------------------

    def train(self):
        """Delegate to BaseTrainer.train()."""
        return self.base.train()
