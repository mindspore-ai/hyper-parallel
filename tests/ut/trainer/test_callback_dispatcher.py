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
"""Source-shape tests for trainer callback dispatcher."""
import ast
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CALLBACK_BASE_PATH = _PROJECT_ROOT / "hyper_parallel/trainer/callbacks/base.py"
_TRAINER_BASE_PATH = _PROJECT_ROOT / "hyper_parallel/trainer/base.py"
_LLM_TRAINER_PATH = _PROJECT_ROOT / "hyper_parallel/trainer/llm_trainer.py"
_VL_TRAINER_PATH = _PROJECT_ROOT / "hyper_parallel/trainer/vl_trainer.py"


def _parse_module(path: Path) -> ast.Module:
    """Parse source without importing backend-dependent modules."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class_node(module: ast.Module, class_name: str) -> ast.ClassDef:
    """Return a class node by name."""
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"Class not found: {class_name}")


def _class_methods(class_node: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    """Return regular methods declared directly on a class."""
    return {
        node.name: node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
    }


def _base_callback_hook_names() -> set[str]:
    """Return public callback hook names from BaseCallback source."""
    callback_module = _parse_module(_CALLBACK_BASE_PATH)
    callback_class = _class_node(callback_module, "BaseCallback")
    return {
        name
        for name in _class_methods(callback_class)
        if name.startswith("on_")
    }


def _callback_hook_members() -> dict[str, str]:
    """Return CallbackHookNames enum members from source."""
    callback_module = _parse_module(_CALLBACK_BASE_PATH)
    callback_hook = _class_node(callback_module, "CallbackHookNames")
    members = {}
    for node in callback_hook.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.startswith("ON_"):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            members[target.id] = node.value.value
    if not members:
        raise AssertionError("CallbackHookNames enum members not found")
    return members


def _base_trainer_methods() -> dict[str, ast.FunctionDef]:
    """Return methods declared directly on BaseTrainer."""
    trainer_module = _parse_module(_TRAINER_BASE_PATH)
    trainer_class = _class_node(trainer_module, "BaseTrainer")
    return _class_methods(trainer_class)


def _call_name(call: ast.Call) -> str:
    """Return best-effort callable name for simple calls."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _function_has_parameter(function_node: ast.FunctionDef, name: str) -> bool:
    """Return whether a function declares a parameter."""
    return any(arg.arg == name for arg in function_node.args.args)


def _passes_keyword_to_call(
        function_node: ast.FunctionDef,
        call_name: str,
        keyword_name: str,
) -> bool:
    """Return whether function calls call_name with keyword_name."""
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node) != call_name:
            continue
        if any(keyword.arg == keyword_name for keyword in node.keywords):
            return True
    return False


def _calls_function(function_node: ast.FunctionDef, call_name: str) -> bool:
    """Return whether a function contains a direct call to call_name."""
    for node in ast.walk(function_node):
        if isinstance(node, ast.Call) and _call_name(node) == call_name:
            return True
    return False


def _dispatch_events(method_node: ast.FunctionDef) -> set[str]:
    """Return event names passed to self._dispatch_callback."""
    hook_members = _callback_hook_members()
    events = set()
    for node in ast.walk(method_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "_dispatch_callback":
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            events.add(first_arg.value)
        elif isinstance(first_arg, ast.Attribute):
            enum_ref = first_arg.value
            if isinstance(enum_ref, ast.Name) and enum_ref.id == "CallbackHookNames":
                events.add(hook_members.get(first_arg.attr, ""))
    events.discard("")
    return events


def _trainer_init(path: Path, class_name: str) -> ast.FunctionDef:
    """Return a trainer class __init__ method."""
    module = _parse_module(path)
    class_node = _class_node(module, class_name)
    return _class_methods(class_node)["__init__"]


class TestTrainerCallbackDispatcher(unittest.TestCase):
    """Validate BaseTrainer callback dispatch surface."""

    def test_hook_names_match_base_callback_hooks(self):
        """CallbackHookNames values stay aligned with BaseCallback hooks."""
        self.assertEqual(set(_callback_hook_members().values()), _base_callback_hook_names())

    def test_base_trainer_implements_all_base_callback_hooks(self):
        """BaseTrainer exposes a facade for every callback hook."""
        trainer_methods = _base_trainer_methods()
        missing = [
            hook_name
            for hook_name in _base_callback_hook_names()
            if hook_name not in trainer_methods
        ]
        self.assertEqual(missing, [])

    def test_base_trainer_hooks_dispatch_to_matching_event(self):
        """Each BaseTrainer hook facade dispatches its matching event."""
        trainer_methods = _base_trainer_methods()
        for hook_name in _base_callback_hook_names():
            with self.subTest(hook_name=hook_name):
                self.assertIn(hook_name, _dispatch_events(trainer_methods[hook_name]))

    def test_base_trainer_receives_callbacks_and_owns_manager(self):
        """BaseTrainer receives callbacks and wires them into CallbackManager."""
        init_method = _base_trainer_methods()["__init__"]
        self.assertTrue(_function_has_parameter(init_method, "callbacks"))
        self.assertTrue(_passes_keyword_to_call(init_method, "CallbackManager", "callbacks"))

    def test_base_trainer_runs_setup_from_init(self):
        """BaseTrainer owns distributed setup during construction."""
        init_method = _base_trainer_methods()["__init__"]
        self.assertTrue(_function_has_parameter(init_method, "setup"))
        self.assertTrue(_calls_function(init_method, "_setup"))

    def test_base_trainer_does_not_construct_concrete_callbacks(self):
        """BaseTrainer should not instantiate concrete callback implementations."""
        trainer_module = _parse_module(_TRAINER_BASE_PATH)
        trainer_class = _class_node(trainer_module, "BaseTrainer")
        constructed_callbacks = []
        for node in ast.walk(trainer_class):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name.endswith("Callback") and name != "BaseCallback":
                constructed_callbacks.append(name)
        self.assertEqual(constructed_callbacks, [])

    def test_base_trainer_owns_checkpoint_lifecycle(self):
        """Checkpoint resume/save should be trainer-owned."""
        train_method = _base_trainer_methods()["train"]
        handle_method = _base_trainer_methods()["_handle_step_control"]
        save_method = _base_trainer_methods()["_save_checkpoint"]
        self.assertTrue(_calls_function(train_method, "_resume_from_checkpoint_if_needed"))
        self.assertTrue(_calls_function(handle_method, "_maybe_save_checkpoint"))
        self.assertTrue(_calls_function(train_method, "_save_final_checkpoint"))
        self.assertTrue(_calls_function(save_method, "_maybe_export_hf_checkpoint"))

    def test_checkpoint_export_callback_classes_are_removed(self):
        """Callback module no longer exposes checkpoint/export mutators."""
        callback_module = _parse_module(_CALLBACK_BASE_PATH)
        callback_classes = {
            node.name
            for node in callback_module.body
            if isinstance(node, ast.ClassDef)
        }
        self.assertNotIn("CheckpointCallback", callback_classes)
        self.assertNotIn("SafetensorsExportCallback", callback_classes)

    def test_wrapped_trainers_forward_user_callbacks(self):
        """Task trainers expose callback injection surface."""
        specs = [
            (_LLM_TRAINER_PATH, "LLMTrainer"),
            (_VL_TRAINER_PATH, "VLTrainer"),
        ]
        for path, class_name in specs:
            with self.subTest(class_name=class_name):
                init_method = _trainer_init(path, class_name)
                self.assertTrue(_function_has_parameter(init_method, "callbacks"))
                self.assertTrue(_passes_keyword_to_call(init_method, "BaseTrainer", "callbacks"))

    def test_wrapped_trainers_build_default_callbacks_via_factory(self):
        """Task trainers inject default callbacks outside BaseTrainer."""
        specs = [
            (_LLM_TRAINER_PATH, "LLMTrainer"),
            (_VL_TRAINER_PATH, "VLTrainer"),
        ]
        for path, class_name in specs:
            with self.subTest(class_name=class_name):
                init_method = _trainer_init(path, class_name)
                self.assertTrue(_calls_function(init_method, "build_default_callbacks"))

    def test_wrapped_trainers_do_not_call_setup(self):
        """BaseTrainer setup is automatic; wrappers should not call it again."""
        specs = [
            (_LLM_TRAINER_PATH, "LLMTrainer"),
            (_VL_TRAINER_PATH, "VLTrainer"),
        ]
        for path, class_name in specs:
            with self.subTest(class_name=class_name):
                init_method = _trainer_init(path, class_name)
                self.assertFalse(_calls_function(init_method, "_setup"))


if __name__ == "__main__":
    unittest.main()
