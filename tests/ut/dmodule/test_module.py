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
"""Unit tests for :mod:`hyper_parallel.dmodule.module`."""

import unittest
from dataclasses import dataclass

from tests.ut.dmodule._ensure_torch_dmodule import ensure_torch_platform_for_dmodule

ensure_torch_platform_for_dmodule()

import torch
from torch import nn

from hyper_parallel.dmodule.model import BaseModel
from hyper_parallel.dmodule.module import Module, ModuleDict, ModuleList, Sequential


class TestModuleInitStates(unittest.TestCase):
    """Tests for Module.init_states behavior."""

    def test_default_init_states_no_param_init_raises(self):
        class SimpleModule(Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(4))

        module = SimpleModule()
        with self.assertRaises(ValueError):
            module.init_states()

    def test_init_states_auto_recurses(self):
        class Child(Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(4))

        class Parent(Module):
            def __init__(self):
                super().__init__()
                self.child = Child()

        module = Parent()
        module.child._param_init = {"weight": nn.init.zeros_}
        module.init_states()
        self.assertTrue(torch.all(module.child.weight == 0))

    def test_init_self_buffers_called(self):
        class BufferModule(Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.zeros(4))
                self.buffer_device_seen = None

            def _init_self_buffers(self, *, buffer_device=None):
                self.buffer_device_seen = buffer_device

        module = BufferModule()
        module.init_states(buffer_device=torch.device("cpu"))
        self.assertEqual(module.buffer_device_seen, torch.device("cpu"))

    def test_no_params_no_error(self):
        class NoParams(Module):
            def __init__(self):
                super().__init__()

        NoParams().init_states()


class TestDiamondInheritance(unittest.TestCase):
    """Diamond inheritance: class Foo(nn.SomeModule, Module)."""

    class TestEmbedding(nn.Embedding, Module):
        def __init__(self, num_embeddings, embedding_dim):
            super().__init__(num_embeddings, embedding_dim)

    def test_module_has_no_init(self):
        self.assertNotIn("__init__", Module.__dict__)

    def test_forward(self):
        emb = self.TestEmbedding(100, 32)
        out = emb(torch.tensor([0, 1, 2]))
        self.assertEqual(out.shape, torch.Size([3, 32]))

    def test_isinstance_checks(self):
        emb = self.TestEmbedding(100, 32)
        self.assertIsInstance(emb, nn.Embedding)
        self.assertIsInstance(emb, nn.Module)
        self.assertIsInstance(emb, Module)


class TestFromNnModule(unittest.TestCase):
    """Tests for Module.from_nn_module utility."""

    def test_is_subclass(self):
        conv2d_cls = Module.from_nn_module(nn.Conv2d)
        self.assertTrue(issubclass(conv2d_cls, nn.Conv2d))
        self.assertTrue(issubclass(conv2d_cls, Module))

    def test_isinstance(self):
        conv2d_cls = Module.from_nn_module(nn.Conv2d)
        module = conv2d_cls(3, 16, 3)
        self.assertIsInstance(module, nn.Conv2d)
        self.assertIsInstance(module, Module)

    def test_init_states_calls_reset_parameters(self):
        layer_norm_cls = Module.from_nn_module(nn.LayerNorm)
        module = layer_norm_cls(32)
        nn.init.zeros_(module.weight)
        module.init_states()
        self.assertTrue(torch.allclose(module.weight, torch.ones(32)))

    def test_init_states_noop_for_parameterless(self):
        gelu_cls = Module.from_nn_module(nn.GELU)
        gelu_cls().init_states()

    def test_cache(self):
        cls1 = Module.from_nn_module(nn.Conv2d)
        cls2 = Module.from_nn_module(nn.Conv2d)
        self.assertIs(cls1, cls2)

    def test_forward_unchanged(self):
        layer_norm_cls = Module.from_nn_module(nn.LayerNorm)
        torch.manual_seed(42)
        orig = nn.LayerNorm(16)
        wrapped = layer_norm_cls(16)
        wrapped.load_state_dict(orig.state_dict())
        x = torch.randn(2, 16)
        torch.testing.assert_close(orig(x), wrapped(x))


class TestContainerInitStates(unittest.TestCase):
    """Tests for ModuleList, ModuleDict, Sequential init_states."""

    def test_module_list_init_states(self):
        layer_norm_cls = Module.from_nn_module(nn.LayerNorm)
        norms = ModuleList([layer_norm_cls(8) for _ in range(3)])
        for norm in norms:
            nn.init.zeros_(norm.weight)
        norms.init_states()
        for norm in norms:
            self.assertTrue(torch.allclose(norm.weight, torch.ones(8)))

    def test_module_dict_init_states(self):
        layer_norm_cls = Module.from_nn_module(nn.LayerNorm)
        norms = ModuleDict({"a": layer_norm_cls(8), "b": layer_norm_cls(8)})
        for norm in norms.values():
            nn.init.zeros_(norm.weight)
        norms.init_states()
        for norm in norms.values():
            self.assertTrue(torch.allclose(norm.weight, torch.ones(8)))

    def test_sequential_init_states(self):
        gelu_cls = Module.from_nn_module(nn.GELU)
        Sequential(gelu_cls()).init_states()

    def test_containers_are_module(self):
        self.assertIsInstance(ModuleList(), Module)
        self.assertIsInstance(ModuleDict(), Module)
        self.assertIsInstance(Sequential(), Module)


class TestConfigBuildPropagatesParamInit(unittest.TestCase):
    """Config.build() propagates param_init to the instance."""

    class ProtocolLinear(Module):
        @dataclass(kw_only=True, slots=True)
        class Config(Module.Config):
            in_features: int = 4
            out_features: int = 4
            param_init: dict | None = None

        def __init__(self, config: "TestConfigBuildPropagatesParamInit.ProtocolLinear.Config"):
            super().__init__()
            self.config = config
            self.weight = nn.Parameter(torch.empty(config.out_features, config.in_features))

    def test_param_init_on_instance(self):
        param_init = {"weight": nn.init.zeros_}
        config = self.ProtocolLinear.Config(
            in_features=4, out_features=4, param_init=param_init
        )
        linear = config.build()
        self.assertIs(linear._param_init, param_init)

    def test_no_param_init_by_default(self):
        linear = self.ProtocolLinear.Config(in_features=4, out_features=4).build()
        self.assertIsNone(linear._param_init)

    def test_init_states_uses_config_param_init(self):
        protocol_linear = self.ProtocolLinear

        class Parent(Module):
            def __init__(self):
                super().__init__()
                linear_config = protocol_linear.Config(
                    in_features=4,
                    out_features=4,
                    param_init={"weight": nn.init.ones_},
                )
                self.linear = linear_config.build()

        module = Parent()
        nn.init.zeros_(module.linear.weight)
        module.init_states()
        self.assertTrue(torch.all(module.linear.weight == 1))


class TestModuleConfigValidation(unittest.TestCase):
    """Module nested configs must use @dataclass(kw_only=True, slots=True)."""

    def test_missing_slots_on_module_config_raises(self):
        with self.assertRaises(TypeError):

            class BadModule(Module):
                @dataclass(kw_only=True)
                class Config(Module.Config):
                    x: int = 1

                def __init__(self, config: "BadModule.Config"):
                    super().__init__()


class TestVerifyModuleProtocol(unittest.TestCase):
    """Tests for BaseModel.verify_module_protocol."""

    class ProtocolLinear(Module):
        @dataclass(kw_only=True, slots=True)
        class Config(Module.Config):
            in_features: int = 4
            out_features: int = 4

        def __init__(self, config: "TestVerifyModuleProtocol.ProtocolLinear.Config"):
            super().__init__()
            self.config = config
            self.weight = nn.Parameter(torch.empty(config.out_features, config.in_features))

    class GoodModel(BaseModel):
        @dataclass(kw_only=True, slots=True)
        class Config(BaseModel.Config):
            def update_from_config(self, *, trainer_config, **kwargs) -> None:
                del trainer_config, kwargs

            def get_nparams_and_flops(self, model: Module, seq_len: int) -> tuple[int, int]:
                del model, seq_len
                return 0, 0

        def __init__(self, protocol_linear_cls):
            super().__init__()
            self.linear = protocol_linear_cls.Config(in_features=4, out_features=4).build()

    class BadModel(BaseModel):
        @dataclass(kw_only=True, slots=True)
        class Config(BaseModel.Config):
            def update_from_config(self, *, trainer_config, **kwargs) -> None:
                del trainer_config, kwargs

            def get_nparams_and_flops(self, model: Module, seq_len: int) -> tuple[int, int]:
                del model, seq_len
                return 0, 0

        def __init__(self):
            super().__init__()
            self.plain = nn.Linear(4, 4)

    def test_passes_for_all_module(self):
        self.GoodModel(self.ProtocolLinear).verify_module_protocol()

    def test_raises_for_plain_nn_module(self):
        with self.assertRaises(RuntimeError):
            self.BadModel().verify_module_protocol()


class TestModuleMisc(unittest.TestCase):
    """Counter / pos-arg cache tests from M1."""

    @dataclass(kw_only=True, slots=True)
    class CounterConfig(Module.Config):
        pass

    class Counter(Module):
        def __init__(self, config: "TestModuleMisc.CounterConfig"):
            super().__init__()
            self.config = config
            self.register_buffer("counter", torch.zeros(1))

        def _init_self_buffers(self, *, buffer_device=None):
            device = buffer_device if buffer_device is not None else torch.device("cpu")
            self.counter = torch.ones(1, device=device)

        def forward(self, x):
            return x

    def test_init_states_initializes_buffers(self):
        module = self.Counter(self.CounterConfig())
        module.init_states(buffer_device=torch.device("cpu"))
        self.assertEqual(module.counter.item(), 1.0)

    def test_cache_pos_arg_names(self):
        module = self.Counter(self.CounterConfig())
        names = module._cache_pos_arg_names()
        self.assertEqual(names, ["x"])
        self.assertIs(module._cache_pos_arg_names(), names)


if __name__ == "__main__":
    unittest.main()
