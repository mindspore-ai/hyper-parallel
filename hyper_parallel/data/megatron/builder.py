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
"""Megatron dataset builder — wires config → IndexedDataset → GPTDataset (+ blend).

Config surface (``args.data.*``):

- ``train_path``: either a single path-prefix (no suffix) pointing to one
  ``.bin``/``.idx`` pair, OR a flat ``"w1 path1 w2 path2 ..."`` string /
  list for :class:`BlendableDataset`.
- ``max_seq_len``: per-sample token count (input plus implicit label shift).
- ``megatron_seed`` (optional, default ``train.seed``): RNG seed for the
  document / sample / shuffle indices.
- ``pad_token_id`` (optional, default 0): right-pad for the tail sample.
- ``eod_token_id`` (optional, default None) + ``eod_mask_loss`` (optional,
  default False): mask EOD tokens out of the loss.

Sample count: ``num_samples = max_steps * global_batch_size`` so the
distributed sampler can drain the dataset without short cycles. Unlike the
HF builder, this builder does NOT modify ``base.state.max_steps``: Megatron
datasets are typically larger than ``num_samples``, so the configured
``max_steps`` is used as-is with no truncation.
"""
import logging
from typing import Any, List, Tuple

from hyper_parallel.data.megatron.blendable_dataset import BlendableDataset
from hyper_parallel.data.megatron.gpt_dataset import GPTDataset
from hyper_parallel.data.megatron.indexed_dataset import IndexedDataset, strip_suffix
from hyper_parallel.data.registry import DATASET_REGISTRY


logger = logging.getLogger(__name__)


def _parse_blend(raw: Any) -> List[Tuple[float, str]]:
    """Parse a blend spec into ``[(weight, prefix), ...]``.

    Supported shapes:

    - ``str`` of whitespace-separated tokens: ``"0.3 /data/a 0.7 /data/b"``
    - ``list[float | str]`` in the same alternating order
    - ``list[list]`` of ``[weight, prefix]`` pairs
    """
    if isinstance(raw, str):
        toks = raw.split()
    elif isinstance(raw, (list, tuple)):
        # Already in pair form?
        if raw and isinstance(raw[0], (list, tuple)):
            return [(float(w), str(p)) for w, p in raw]
        toks = list(raw)
    else:
        raise ValueError(f"Unsupported blend spec type: {type(raw)}")

    if len(toks) % 2 != 0:
        raise ValueError(
            f"Blend spec must have an even number of tokens "
            f"(alternating weight + prefix); got {len(toks)}: {toks}"
        )
    out: List[Tuple[float, str]] = []
    for i in range(0, len(toks), 2):
        out.append((float(toks[i]), str(toks[i + 1])))
    return out


def _looks_like_blend(train_path: Any) -> bool:
    """Return ``True`` when ``train_path`` is a multi-source blend spec.

    A blend spec always starts with a numeric weight (``"0.3 /a 0.7 /b"`` or
    a list of pairs); a single corpus is a bare path-prefix. Keying off "the
    first token parses as a float" (rather than "contains whitespace") lets a
    single path that legitimately contains spaces still be read as one source.
    """
    if isinstance(train_path, (list, tuple)):
        return True
    if not isinstance(train_path, str):
        return False
    toks = train_path.split()
    if len(toks) < 2:
        return False
    try:
        float(toks[0])
    except ValueError:
        return False
    return True


@DATASET_REGISTRY.register("megatron")
def build_megatron(*, base: Any, args: Any, **_: Any) -> Any:
    """Build a Megatron ``.bin``/``.idx`` dataset (single source or blend)."""
    del base
    data_cfg = args.data
    train_cfg = args.train
    train_path = data_cfg.train_path
    if not train_path:
        raise ValueError("data.train_path is required when data.type='megatron'")

    seq_length = int(data_cfg.max_seq_len)
    # ``megatron_seed`` may be present-but-None on a default DataConfig; fall
    # through to ``train.seed`` in that case so the typed config doesn't have
    # to invent a sentinel.
    megatron_seed = data_cfg.megatron_seed
    if megatron_seed is None:
        megatron_seed = train_cfg.seed
    seed = int(megatron_seed)
    pad_token_id = int(data_cfg.pad_token_id)
    eod_token_id = data_cfg.eod_token_id
    eod_mask_loss = bool(data_cfg.eod_mask_loss)
    mmap_bin = bool(data_cfg.mmap_bin_files)

    num_samples = int(train_cfg.max_steps * train_cfg.global_batch_size)
    if num_samples <= 0:
        raise ValueError(
            f"num_samples = max_steps * global_batch_size must be > 0; "
            f"got max_steps={train_cfg.max_steps}, global_bs={train_cfg.global_batch_size}"
        )

    # Single source: simplest path, no blend needed.
    if not _looks_like_blend(train_path):
        prefix = strip_suffix(train_path)
        idx_ds = IndexedDataset(prefix, mmap=mmap_bin)
        gpt_ds = GPTDataset(
            idx_ds, num_samples=num_samples, seq_length=seq_length,
            seed=seed, pad_token_id=pad_token_id,
            eod_mask_loss=eod_mask_loss, eod_token_id=eod_token_id,
        )
        logger.info(
            "Megatron dataset built: prefix=%s docs=%d sequences=%d samples=%d seq_len=%d",
            prefix, idx_ds.num_documents, len(idx_ds), len(gpt_ds), seq_length,
        )
        return gpt_ds

    # Blend: parse, build per-source GPTDataset, wrap in BlendableDataset.
    pairs = _parse_blend(train_path)
    weights = [w for w, _ in pairs]
    sub_datasets = []
    for w, p in pairs:
        prefix = strip_suffix(p)
        idx_ds = IndexedDataset(prefix, mmap=mmap_bin)
        # Each sub-dataset is sized to cover its share of the blend; we ask
        # for the full ``num_samples`` here because blending picks indices
        # modulo the sub-dataset length.
        sub_datasets.append(
            GPTDataset(
                idx_ds, num_samples=num_samples, seq_length=seq_length,
                seed=seed, pad_token_id=pad_token_id,
                eod_mask_loss=eod_mask_loss, eod_token_id=eod_token_id,
            )
        )
        logger.info(
            "  blend source w=%g prefix=%s docs=%d sequences=%d",
            w, prefix, idx_ds.num_documents, len(idx_ds),
        )
    return BlendableDataset(sub_datasets, weights, num_samples=num_samples, seed=seed)
