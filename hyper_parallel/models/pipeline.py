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
"""Model-owned pipeline partitioning before parameter sharding and loading."""

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from torch import distributed as dist, nn

from hyper_parallel.models.registry import resolve_model_adapter


@dataclass
class PipelineConfig:
    """Configure local model stages and their scheduling on each PP rank.

    Attributes:
        schedule: Microbatch forward/backward order.
        split_points: Global decoder indices starting stages after stage zero.
            An empty list selects an even contiguous partition.
        virtual_stages_per_rank: Number of round-robin chunks on each physical stage.
            Defaults to two for interleaved 1F1B and MPipe, and one for 1F1B and GPipe.
    """

    schedule: Literal["1f1b", "gpipe", "interleaved_1f1b", "mpipe"] = "1f1b"
    split_points: list[int] = field(default_factory=list)
    virtual_stages_per_rank: Optional[int] = None

    def __post_init__(self) -> None:
        if self.schedule not in ("1f1b", "gpipe", "interleaved_1f1b", "mpipe"):
            raise ValueError("pipeline.schedule must be '1f1b', 'gpipe', 'interleaved_1f1b' or 'mpipe'")
        count = self.local_stage_count
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("pipeline.virtual_stages_per_rank must be a positive integer")
        if count > 1 and self.schedule not in ("interleaved_1f1b", "mpipe"):
            raise ValueError("multiple virtual stages per rank require interleaved_1f1b or mpipe")
        if any(not isinstance(point, int) or isinstance(point, bool) for point in self.split_points):
            raise TypeError("pipeline.split_points must contain integer layer indices")

    @property
    def local_stage_count(self) -> int:
        """Resolve the schedule default after YAML and CLI overrides."""
        if self.virtual_stages_per_rank is None:
            return 2 if self.schedule in ("interleaved_1f1b", "mpipe") else 1
        return self.virtual_stages_per_rank


@dataclass(frozen=True)
class PipelinePartition:
    """Global layer ownership and stage position, independent of model family."""

    stage_index: int
    stage_count: int
    start: int
    stop: int
    mpipe: bool = False
    preprocess_replica: bool = False

    @property
    def is_first(self) -> bool:
        """Whether this stage owns the embedding input."""
        return self.stage_index == 0

    @property
    def is_last(self) -> bool:
        """Whether this stage owns the output objective."""
        return self.stage_index == self.stage_count - 1


@dataclass(frozen=True)
class PipelineAdapterSpec:
    """Locate decoder layers and install the family's stage execution contract.

    The installer prunes endpoint modules and accepts ``pipeline_inputs`` as a
    tuple of received tensors. Nonterminal stages return a tensor tuple; the
    terminal stage preserves the model's ordinary output/loss interface.
    """

    layers_fqn: str
    install: Callable[[nn.Module, PipelinePartition], None]
    combine_chunks: Optional[Callable[[tuple[nn.Module, ...], dist.ProcessGroup], nn.Module]] = None


def partition_model(model: nn.Module, mesh: object, config: Optional[PipelineConfig] = None,
                    chunk_index: int = 0) -> None:
    """Keep local layers with global FQNs before any weights are materialized.

    Args:
        model: Constructed model, normally on meta.
        mesh: Existing distributed MeshContext.
        config: Optional explicit cut points and schedule selection.
        chunk_index: Local virtual chunk in round-robin pipeline order.
    """
    pp_size = getattr(mesh, "pp_size", 1)
    if pp_size == 1:
        return
    config = config or PipelineConfig()
    stage_count = pp_size * config.local_stage_count
    if not 0 <= chunk_index < config.local_stage_count:
        raise ValueError("local pipeline chunk index is outside the configured partition")
    architecture = (getattr(model.config, "architectures", None) or [""])[0]
    adapter = resolve_model_adapter(model)
    provider = getattr(adapter, "pipeline_parallel", None)
    if provider is None:
        raise ValueError(f"{architecture or model.config.model_type} has no pipeline-parallel adapter")
    spec = provider()
    if not isinstance(spec, PipelineAdapterSpec):
        raise TypeError("pipeline_parallel provider must return PipelineAdapterSpec")
    if getattr(model.config, "tie_word_embeddings", False):
        raise ValueError("Trainer pipeline stages currently require untied input/output embeddings")
    layers = model.get_submodule(spec.layers_fqn)
    layer_count = len(layers)
    if stage_count > layer_count:
        raise ValueError("pipeline stages must not outnumber decoder layers")
    points = config.split_points or [layer_count * index // stage_count for index in range(1, stage_count)]
    boundaries = [0, *points, layer_count]
    if len(points) != stage_count - 1 or any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("pipeline.split_points must define one nonempty ordered layer range per pipeline stage")
    rank = mesh.pp_rank + chunk_index * pp_size
    partition = PipelinePartition(rank, stage_count, boundaries[rank], boundaries[rank + 1],
                                  mpipe=config.schedule == "mpipe",
                                  preprocess_replica=config.schedule == "mpipe" and chunk_index == 0)
    # Install while the complete tree is available for cross-layer dependency analysis.
    spec.install(model, partition)
    parent_fqn, name = spec.layers_fqn.rsplit(".", 1)
    parent = model.get_submodule(parent_fqn)
    local_layers = nn.ModuleDict({str(index): layers[index] for index in range(partition.start, partition.stop)})
    setattr(parent, name, local_layers)
    model._hp_pipeline_partition = partition  # pylint: disable=protected-access


__all__ = ["PipelineAdapterSpec", "PipelineConfig", "PipelinePartition", "partition_model"]
