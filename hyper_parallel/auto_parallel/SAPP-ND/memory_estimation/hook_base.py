# Copyright 2025 Huawei Technologies Co., Ltd
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
"""hook base module"""
from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABCMeta
from abc import abstractmethod
import inspect

if TYPE_CHECKING:
    from typing import Dict, Callable


class HookMetaclass(ABCMeta):
    """meta class"""

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        hook_registry = getattr(cls, "hook_registry", {})
        # For each abstract method in base classes, compare signatures
        for base in bases:
            for meth in getattr(base, "__abstractmethods__", set()):
                base_method = getattr(base, meth)
                sub_method = getattr(cls, meth, None)
                base_sig = inspect.signature(base_method)
                sub_sig = inspect.signature(sub_method)
                base_params = list(base_sig.parameters.values())
                sub_params = list(sub_sig.parameters.values())
                if getattr(sub_method, "__isabstractmethod__", False):
                    raise TypeError(
                        f"{meth}() abstract method needs "
                        f"to be implemented in subclass"
                    )
                if sub_method not in hook_registry.values():
                    raise TypeError(
                        f"{meth}() needs to be decorated with "
                        f"@hook_runner(model_name), "
                        f"with model_name as a string"
                    )
                if len(base_params) != len(sub_params):
                    raise TypeError(
                        f"Mismatch signature of {meth}() "
                        f"in class {cls.__name__}: "
                        f"Expected {[p.name for p in base_params]} "
                        f"instead of {[p.name for p in sub_params]})"
                    )


def hook_runner(model_name: str) -> Callable:
    """decorator"""
    if not model_name or not isinstance(model_name, str):
        raise TypeError(
            f"@hook_runner decorator " f"has invalid model_name ({model_name})"
        )
    if (
        model_name in MemEvalHook.hook_registry
    ):  # pylint: disable=protected-access
        raise TypeError(
            f"@hook_runner model_name " f"'{model_name}' is already defined"
        )

    def wrapper(func):
        """register"""
        MemEvalHook.hook_registry[model_name] = (
            func  # pylint: disable=protected-access
        )
        return func

    return wrapper


class MemEvalHook(metaclass=HookMetaclass):
    """abstract base hook class"""

    hook_registry = {}

    def get_hooks(self) -> Dict:
        """gather all the child class hooks"""
        target_cls = [self.__class__] + list(self.__class__.__bases__)
        cls_names = [c.__name__ for c in target_cls]
        res = {
            k: v
            for k, v in MemEvalHook.hook_registry.items()
            if v.__qualname__.split(".")[-2] in cls_names
        }
        # print(self.__class__.__name__)
        # print(self.__class__.__subclasses__())
        # print(MemEvalHook.__subclasses__())
        return res

    @staticmethod
    @abstractmethod
    def run_hooks(e):
        """must implement"""
        pass  # pylint: disable=unnecessary-pass


# Testing

# class A(MemEvalHook) :
#     @hook_runner("a")
#     def run_hooks(e) :
#         print("a stuff")

#     @hook_runner("a2")
#     def run_hooks(e) :
#         print("a2 stuff")

# class B(MemEvalHook) :
#     @hook_runner("b")
#     def run_hooks(e) :
#         print("b stuff")

# class C(A,B) : pass

# k = C()
# h = k.get_hooks()
# print(h)
# for r in h.values() : r(None)
# k = A()
# h = k.get_hooks()
# print(h)
# for r in h.values() : r(None)

# class D(A) : pass
# k = D()
# h = k.get_hooks()
# print(h)
# for r in h.values() : r(None)

# class E() :
#     def a() : print(1)
#     def a() : print(2)
# E.a()
