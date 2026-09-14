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
"""Configuration dataclasses for Dense LLM strategy tuner."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import yaml


@dataclass
class HardwareConfig:
    """Hardware configuration for strategy tuning.

    Attributes:
        num_devices: Total number of devices available.
        device_type: Device type name, e.g. "A2", "A3", "V100".
        memory_per_device_mb: Memory capacity per device in MB.
        intra_node_bandwidth_gb: Intra-node bandwidth in GB/s.
        inter_node_bandwidth_gb: Inter-node bandwidth in GB/s.
        devices_per_node: Number of devices per node.
        tflops_per_device: Peak compute throughput per device in TFLOPS.
    """

    num_devices: int = 8
    device_type: str = "A2"
    memory_per_device_mb: float = 65536.0
    intra_node_bandwidth_gb: float = 50.0
    inter_node_bandwidth_gb: float = 10.0
    devices_per_node: int = 8
    tflops_per_device: float = 300.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HardwareConfig:
        """Create from a dictionary.

        Args:
            data: Dictionary with hardware configuration.

        Returns:
            HardwareConfig instance.
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})  # pylint: disable=no-member

    @classmethod
    def from_yaml(cls, path: str) -> HardwareConfig:
        """Create from a YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            HardwareConfig instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            yaml.YAMLError: If the YAML content is invalid.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        hw = data.get("hardware", data)
        return cls.from_dict(hw)

    def to_sapp_machine(self) -> Any:
        """Convert to an SAPP-ND ``Machine`` object.

        Returns:
            A ``Machine(number, device)`` instance suitable for
            ``ParallelizeLayer`` or ``Parallelize``.

        Example::

            machine = hardware.to_sapp_machine()
            pl = ParallelizeLayer(evaluator, machine)
        """
        # pylint: disable=import-outside-toplevel
        from hyper_parallel.auto_parallel.sapp_nd.nd.common import hardware as Hard

        return Hard.Machine(number=self.num_devices, device=self.device_type)


@dataclass
class ModelConfig:
    """Dense LLM model configuration for strategy tuning.

    Attributes:
        model_name: Model name, e.g. "llama-7b".
        hidden_size: Hidden dimension size.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        num_kv_heads: Number of KV heads (for GQA). 0 means same as num_heads.
        intermediate_size: FFN intermediate dimension. 0 means 4 * hidden_size.
        vocab_size: Vocabulary size.
        seq_length: Sequence length.
        precision_bytes: Bytes per parameter (2 for bf16, 4 for fp32).
        tie_word_embeddings: Whether input and output embeddings share weights.
    """

    model_name: str = "llama-7b"
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 0
    intermediate_size: int = 0
    vocab_size: int = 32000
    seq_length: int = 4096
    precision_bytes: int = 2
    tie_word_embeddings: bool = True

    @property
    def effective_num_kv_heads(self) -> int:
        """Number of KV heads, defaulting to num_heads if not set."""
        return self.num_kv_heads if self.num_kv_heads > 0 else self.num_heads

    @property
    def effective_intermediate_size(self) -> int:
        """FFN intermediate size, defaulting to 4 * hidden_size if not set."""
        return self.intermediate_size if self.intermediate_size > 0 else 4 * self.hidden_size

    @property
    def num_params_billion(self) -> float:
        """Estimated number of parameters in billions."""
        h = self.hidden_size
        hff = self.effective_intermediate_size
        v = self.vocab_size
        n = self.num_layers
        a = self.num_heads
        kv = self.effective_num_kv_heads
        emb = v * h
        attn = 2 * (1 + kv / a) * h * h
        ffn = 3 * h * hff
        norm = 2 * 2 * h * n
        output_emb = 0 if self.tie_word_embeddings else v * h
        total = emb + n * (attn + ffn) + norm + output_emb
        return total / 1e9

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelConfig:
        """Create from a dictionary.

        Args:
            data: Dictionary with model configuration.

        Returns:
            ModelConfig instance.
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})  # pylint: disable=no-member

    @classmethod
    def from_yaml(cls, path: str) -> ModelConfig:
        """Create from a YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            ModelConfig instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            yaml.YAMLError: If the YAML content is invalid.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        model = data.get("model", data)
        return cls.from_dict(model)


@dataclass
class SearchSpaceConfig:
    """User-defined search space constraints for parallel dimensions.

    Attributes:
        dp_range: Allowed DP degrees. Empty means auto-enumerate.
        tp_range: Allowed TP degrees. Empty means auto-enumerate.
        pp_range: Allowed PP degrees. Empty means auto-enumerate.
        cp_range: Allowed CP degrees. Empty means [1] (disabled).
        micro_batch_num_range: Allowed micro-batch numbers. Empty means auto.
        enable_cp: Whether to include context parallelism in search.
        enable_pp: Whether to include pipeline parallelism in search.
    """

    dp_range: List[int] = field(default_factory=list)
    tp_range: List[int] = field(default_factory=list)
    pp_range: List[int] = field(default_factory=list)
    cp_range: List[int] = field(default_factory=list)
    micro_batch_num_range: List[int] = field(default_factory=list)
    enable_cp: bool = False
    enable_pp: bool = True
    enable_fsdp: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SearchSpaceConfig:
        """Create from a dictionary.

        Args:
            data: Dictionary with search space configuration.

        Returns:
            SearchSpaceConfig instance.
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})  # pylint: disable=no-member

    @classmethod
    def from_yaml(cls, path: str) -> SearchSpaceConfig:
        """Create from a YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            SearchSpaceConfig instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            yaml.YAMLError: If the YAML content is invalid.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        ss = data.get("search_space", data)
        return cls.from_dict(ss)


@dataclass
class ConstraintConfig:
    """Constraints for filtering candidate strategies.

    Attributes:
        memory_limit_mb: Maximum memory per device in MB. 0 means unlimited.
        global_batch_size: Required global batch size. 0 means auto.
        min_dp_degree: Minimum DP degree required.
        max_step_time_ms: Maximum acceptable step time in ms. 0 means unlimited.
    """

    memory_limit_mb: float = 0.0
    global_batch_size: int = 0
    min_dp_degree: int = 1
    max_step_time_ms: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConstraintConfig:
        """Create from a dictionary.

        Args:
            data: Dictionary with constraint configuration.

        Returns:
            ConstraintConfig instance.
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})  # pylint: disable=no-member

    @classmethod
    def from_yaml(cls, path: str) -> ConstraintConfig:
        """Create from a YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            ConstraintConfig instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            yaml.YAMLError: If the YAML content is invalid.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cc = data.get("constraints", data)
        return cls.from_dict(cc)


@dataclass
class TunerConfig:
    """Top-level configuration for the Dense LLM strategy tuner.

    Attributes:
        hardware: Hardware configuration.
        model: Model configuration.
        search_space: Search space constraints.
        constraints: Feasibility constraints.
        top_k: Number of top strategies to return.
        ppb_top_k: Number of best strategies to run pipeline balancing
            (SAPP-PPB) on. Kept small to avoid repeatedly invoking the
            expensive PPB solver for the whole candidate set.
        sort_by: Sort criterion, e.g. "step_time" or "memory".
        output_dir: Directory for output files and visualizations.
        verbose: Whether to print detailed logs.
    """

    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    search_space: SearchSpaceConfig = field(default_factory=SearchSpaceConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    top_k: int = 5
    ppb_top_k: int = 5
    sort_by: str = "step_time"
    output_dir: str = "./tuner_output"
    verbose: bool = False
    profiling_json_path: str = ""

    @classmethod
    def from_yaml(cls, path: str) -> TunerConfig:
        """Create from a single YAML configuration file.

        Args:
            path: Path to YAML file containing all config sections.

        Returns:
            TunerConfig instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            yaml.YAMLError: If the YAML content is invalid.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        hw = HardwareConfig.from_dict(data.get("hardware", {}))
        model = ModelConfig.from_dict(data.get("model", {}))
        ss = SearchSpaceConfig.from_dict(data.get("search_space", {}))
        cc = ConstraintConfig.from_dict(data.get("constraints", {}))
        top_k = data.get("top_k", 5)
        ppb_top_k = data.get("ppb_top_k", top_k)
        sort_by = data.get("sort_by", "step_time")
        output_dir = data.get("output_dir", "./tuner_output")
        verbose = data.get("verbose", False)
        return cls(
            hardware=hw,
            model=model,
            search_space=ss,
            constraints=cc,
            top_k=top_k,
            ppb_top_k=ppb_top_k,
            sort_by=sort_by,
            output_dir=output_dir,
            verbose=verbose,
            profiling_json_path=data.get("profiling_json_path", ""),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TunerConfig:
        """Create from a dictionary.

        Args:
            data: Dictionary with tuner configuration.

        Returns:
            TunerConfig instance.
        """
        hw = HardwareConfig.from_dict(data.get("hardware", {}))
        model = ModelConfig.from_dict(data.get("model", {}))
        ss = SearchSpaceConfig.from_dict(data.get("search_space", {}))
        cc = ConstraintConfig.from_dict(data.get("constraints", {}))
        return cls(
            hardware=hw,
            model=model,
            search_space=ss,
            constraints=cc,
            top_k=data.get("top_k", 5),
            ppb_top_k=data.get("ppb_top_k", data.get("top_k", 5)),
            sort_by=data.get("sort_by", "step_time"),
            output_dir=data.get("output_dir", "./tuner_output"),
            verbose=data.get("verbose", False),
            profiling_json_path=data.get("profiling_json_path", ""),
        )

    def create_sapp_parallelize(self) -> Any:
        """Create a SAPP-ND ``Parallelize`` instance from this config.

        Generates a Hyper V2 YAML configuration, builds a ``Machine``,
        and instantiates ``Parallelize(framework="hyper_v2",
        config=yaml_path, machine)`` so that ND handles ccfg
        construction and parser initialization internally.

        The ``hyper_v2`` framework uses ``CostModelParserHyperV2``
        which parses the HyperParallel-native YAML format
        (``model.name`` + ``model.config_overrides``,
        ``train.accelerator``, ``data.max_seq_len``).

        Returns:
            A ``Parallelize`` instance with a properly-initialised
            ``ParallelizeLayer`` (accessible via ``__getattr__``).

        Example::

            par = cfg.create_sapp_parallelize()
            scored_space = par.run_generation_to_ordering("")
        """
        # pylint: disable=import-outside-toplevel,cyclic-import
        from hyper_parallel.auto_parallel.dense_tuner.candidate_generator import (
            _build_sapp_dimensions,
        )
        from hyper_parallel.auto_parallel.sapp_nd.nd.logger import set_verbose_level
        from hyper_parallel.auto_parallel.sapp_nd.nd.parallelize import Parallelize

        set_verbose_level(1)

        yaml_path = self.to_sapp_yaml()
        machine = self.hardware.to_sapp_machine()
        dimensions = _build_sapp_dimensions(self.search_space)
        extra: Dict[str, Any] = {}
        if self.constraints.global_batch_size > 0:
            extra["global_batch_size"] = self.constraints.global_batch_size
        if dimensions is not None:
            extra["dimensions"] = dimensions
        par = Parallelize("hyper_v2", yaml_path, machine, **extra)
        ccfg = par.config.ccfg
        if not hasattr(ccfg, "vp") or ccfg.vp == 0:
            ccfg.vp = 1
        par._yaml_path = yaml_path
        return par

    def to_sapp_yaml(self) -> str:
        """Write a Hyper V2 YAML configuration for SAPP-ND ``Parallelize``.

        Generates a YAML file in the HyperParallel-native format that
        ``CostModelParserHyperV2`` expects::

            model:
              name: <model_name>
              config_overrides:
                hidden_size: ...
                num_hidden_layers: ...
                ...
            train:
              accelerator: ...
              gradient_checkpointing: ...
              global_batch_size: ...
              micro_batch_size: ...
            data:
              max_seq_len: ...
            context:
              max_device_memory: ...

        Returns:
            Path to the generated temporary YAML file.

        Example::

            yaml_path = tuner_cfg.to_sapp_yaml()
            par = Parallelize("hyper_v2", yaml_path, machine)
        """
        # pylint: disable=import-outside-toplevel
        import tempfile

        m = self.model
        h = self.hardware
        c = self.constraints

        dtype_str = "bfloat16" if m.precision_bytes == 2 else "float32"
        param_dtype_str = "float32" if m.precision_bytes == 4 else "float32"
        mem_gb = int(h.memory_per_device_mb / 1024)

        ac_mode = "full"
        if hasattr(self, "_recompute_mode"):
            ac_mode = self._recompute_mode

        doc = {
            "model": {
                "name": m.model_name,
                "config_overrides": {
                    "hidden_size": m.hidden_size,
                    "num_hidden_layers": m.num_layers,
                    "num_attention_heads": m.num_heads,
                    "num_key_value_heads": m.effective_num_kv_heads,
                    "intermediate_size": m.effective_intermediate_size,
                    "vocab_size": m.vocab_size,
                    "max_position_embeddings": m.seq_length,
                    "tie_word_embeddings": m.tie_word_embeddings,
                    "mtp_depth": 0,
                },
                "param_init_type": param_dtype_str,
                "compute_dtype": dtype_str,
                "softmax_compute_type": "float32",
            },
            "data": {
                "max_seq_len": m.seq_length,
            },
            "train": {
                "global_batch_size": max(1, c.global_batch_size),
                "micro_batch_size": 1,
                "accelerator": {
                    "dp_replicate": 1,
                    "dp_shard": 1,
                    "tp_degree": 1,
                    "pipeline_parallel_degree": 1,
                    "context_parallel_degree": 1,
                    "expert_parallel_degree": 1,
                    "pp_interleave_num": 1,
                    "use_seq_parallel": True,
                    "enable_parallel_optimizer": c.global_batch_size > 0,
                    "pipeline_scheduler": "1f1b",
                },
                "gradient_checkpointing": {
                    "activation_checkpoint": ac_mode,
                },
                "optimizer": {
                    "type": "adamw",
                    "max_grad_norm": 0.0,
                },
            },
            "context": {
                "max_device_memory": f"{mem_gb}GB",
            },
        }

        fd, path = tempfile.mkstemp(suffix=".yaml", prefix="dense_tuner_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.dump(doc, f, default_flow_style=False, allow_unicode=True)

        return path
