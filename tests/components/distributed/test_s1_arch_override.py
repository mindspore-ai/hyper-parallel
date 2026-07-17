# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S1.2: ARCH_OVERRIDES 覆盖优先级 + _get_architecture。"""

import torch.nn as nn

from hyper_models.components.distributed.param_role import (
    ParameterClassifier,
    ParamRole,
)
from hyper_models.components.distributed.sharding_planner import ShardingPlanner


class _Model(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(8, 4)
        self.output_head = nn.Linear(4, 8, bias=False)


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class TestArchOverridePriority:
    def test_override_beats_default(self):
        """override 命中 → 覆盖默认规则（embed_tokens 默认 EMBED，强制为 SKIP）。"""
        overrides = {"myarch": [("embed_tokens.weight", ParamRole.SKIP)]}
        clf = ParameterClassifier(arch_overrides=overrides)
        model = _Model()
        roles = clf.classify(model, "myarch")
        assert roles["token_embed.weight"] == ParamRole.SKIP

    def test_override_list_of_patterns(self):
        """list-of-patterns 写法：任一子模式命中即覆盖。"""
        overrides = {"myarch": [(["token_embed", "word_embed"], ParamRole.EMBED)]}
        clf = ParameterClassifier(arch_overrides=overrides)
        roles = clf.classify(_Model(), "myarch")
        assert roles["token_embed.weight"] == ParamRole.EMBED

    def test_default_when_no_override_match(self):
        """override 未命中 → 默认规则（output_head 非标准名 → SKIP）。"""
        overrides = {"myarch": [("token_embed", ParamRole.EMBED)]}
        clf = ParameterClassifier(arch_overrides=overrides)
        roles = clf.classify(_Model(), "myarch")
        assert roles["output_head.weight"] == ParamRole.SKIP

    def test_unknown_arch_falls_back_to_default(self):
        overrides = {"other": [("token_embed", ParamRole.EMBED)]}
        clf = ParameterClassifier(arch_overrides=overrides)
        roles = clf.classify(_Model(), "myarch")
        assert roles["token_embed.weight"] == ParamRole.SKIP


class TestGetArchitecture:
    def setup_method(self):
        self.planner = ShardingPlanner()

    def test_architectures_first(self):
        m = _Model(config=_Cfg(architectures=["Qwen2ForCausalLM"], model_type="qwen2"))
        assert self.planner._get_architecture(m) == "qwen2"

    def test_model_type_fallback(self):
        m = _Model(config=_Cfg(architectures=None, model_type="mixtral"))
        assert self.planner._get_architecture(m) == "mixtral"

    def test_classname_fallback(self):
        class LlamaForCausalLM(nn.Module):
            config = None
        assert self.planner._get_architecture(LlamaForCausalLM()) == "llama"

    def test_suffix_stripping(self):
        for cls_name, want in [
            ("LlamaForCausalLM", "llama"),
            ("Blip2ForConditionalGeneration", "blip2"),
            ("BertForSequenceClassification", "bert"),
            ("PaliGemmaForImageTextToText", "paligemma"),
        ]:
            cls = type(cls_name, (nn.Module,), {"config": None})
            assert self.planner._get_architecture(cls()) == want

    def test_no_config_attribute(self):
        class Tiny(nn.Module):
            pass
        assert self.planner._get_architecture(Tiny()) == "tiny"
