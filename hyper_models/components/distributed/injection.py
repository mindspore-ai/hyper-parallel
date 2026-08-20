# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""injection: 注入函数的模板装饰器与签名校验（显式注入机制的入口纪律）。

设计原则：**注入函数 = apply 期被调用一次、必选声明 mesh 家族（框架
按名填充，未激活轴为 None）、用不用随你**。两个装饰器覆盖全部注入
通道，形态各自唯一（2026-08-10 起 local_compute_fn 的直传 compute fn
形态退役，统一为工厂形态——与 inner_wrapper 同一规范：mesh 家族永远
传入，是否使用是用户自己的选择）：

- ``@local_compute``：区域计算**工厂**，``fn(mesh, tp_mesh, cp_mesh,
  ep_mesh, [module], <配置键...>) -> compute_fn``——apply 时 build 一次
  （典型：从 ep_mesh 建通信组闭包固定，运行时零 mesh 开销），返回的
  compute fn ``fn(module, *原forward入参)`` 无需再装饰（签名校验见下）；
- ``@inner_wrapper``：inner forward 包装，``fn(target_module, mesh,
  tp_mesh, cp_mesh, ep_mesh) -> None``——原地替换 target.forward。

框架上下文全集（保留名，语义由框架定义，不可被用户配置）：
- 锚点：``target_module``（inner_wrapper 被包装的模块，必选）/
  ``module``（@local_compute 的边界模块，可选声明）——注入的作用对象；
- mesh 家族（**必选**，四类全声明、全由框架填充；对应轴未激活时为
  None）：``mesh``（当前 plan 坐标系的 active DTensor mesh，dp 已剥离，
  与 PrecompiledBoundary / resolve_placements 的坐标系一致）、
  ``tp_mesh`` / ``cp_mesh`` / ``ep_mesh``（D-10 TP-extend-EP 派生的
  (edp, ep) expert mesh——它同时是专家参数的分片域，由框架统一派生，
  保证 a2a 通信域与分片域严格一致）；

装饰器的硬性规则（import 期即 fail-fast）：
- 必选上下文缺一不可：两个装饰器都必须声明
  ``mesh``/``tp_mesh``/``cp_mesh``/``ep_mesh``；``@inner_wrapper`` 在此
  之上还必须声明 ``target_module``；
- 上下文参数**不得有默认值**（框架必然填充，默认值无意义），用户在
  Target/YAML 里配置同名保留键 → fail-fast（保留名不可配置）；
- 注入函数禁止 ``*args`` / ``**kwargs``——签名必须是显式形参列表，这
  是"配置键按名绑定、拼写错误不得被静默吞掉"制度的前提；
- 其余具名参数是**用户配置键**，全部来自 YAML/Target 的显式配置，框
  架不做任何自动填充；配置键只接受数据值——**不允许再往注入函数里传
  函数**（函数套函数无穷无尽；需要自定义行为就写自己的注入函数，路由/
  排布等逻辑直接写死在函数体内）。

运行时层校验（apply 时 fail-fast）：
- ``validate_local_compute_signature``: 工厂返回的 compute fn 的每个
  入参必须在原 forward 中有同名形参、位置序一致、forward 的必填参数
  必须全部被接住——"注入函数的入参与原函数匹配"；
- ``validate_wrapped_forward``: inner_wrapper 替换后的 forward 必须能绑
  定原 forward 的全部入参（dummy bind 试探；替换侧允许 *args/**kwargs
  宽容透传）。
"""

import inspect
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 注入种类（meta.kind）
LOCAL_COMPUTE = "local_compute"        # 区域计算工厂（local_compute_fn 唯一形态）
INNER_WRAPPER = "inner_wrapper"        # inner forward 包装

# 各种类的框架上下文保留名（声明即由框架按名填充；配置同名键报错）
_MESH_FAMILY = frozenset({"mesh", "tp_mesh", "cp_mesh", "ep_mesh"})
FACTORY_CONTEXT = frozenset({"module"}) | _MESH_FAMILY
WRAPPER_CONTEXT = frozenset({"target_module"}) | _MESH_FAMILY

# 各种类的必选上下文（装饰器强制声明齐全——mesh 家族全由框架传入，
# 用户只管使用；对应轴未激活时框架填 None）
FACTORY_REQUIRED = _MESH_FAMILY
WRAPPER_REQUIRED = frozenset({"target_module"}) | _MESH_FAMILY

_ALLOWED_CONTEXT = {
    LOCAL_COMPUTE: FACTORY_CONTEXT,    # module 锚点可选 + mesh 家族
    INNER_WRAPPER: WRAPPER_CONTEXT,
}
_REQUIRED_CONTEXT = {
    LOCAL_COMPUTE: FACTORY_REQUIRED,   # mesh 家族必选
    INNER_WRAPPER: WRAPPER_REQUIRED,
}

_DECORATOR_NAMES = {
    LOCAL_COMPUTE: "@local_compute",
    INNER_WRAPPER: "@inner_wrapper",
}


@dataclass(frozen=True)
class InjectionMeta:
    """装饰器写在注入函数对象上的元数据（``fn._injection_meta``）。"""
    kind: str                 # LOCAL_COMPUTE / INNER_WRAPPER
    context: frozenset        # 声明的框架上下文键


def _make_decorator(kind):
    allowed = _ALLOWED_CONTEXT[kind]
    required = _REQUIRED_CONTEXT[kind]

    def decorator(fn):
        fname = getattr(fn, "__name__", fn)
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{_DECORATOR_NAMES[kind]}: 无法内省 {fn!r} 的签名（注入函数"
                "必须是可内省的普通 callable）") from exc
        context = []
        for name, p in sig.parameters.items():
            if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD):
                raise TypeError(
                    f"{_DECORATOR_NAMES[kind]} 注入函数 {fname}"
                    f" 不允许 *args/**kwargs（形参 {name!r}）——注入函数的签名"
                    "必须是显式形参列表：上下文按名声明由框架填充，配置键按名"
                    "绑定，**kwargs 会把拼写错误静默吞掉（与 "
                    "_check_target_config_keys 制度冲突）")
            if name in allowed:
                if p.default is not inspect.Parameter.empty:
                    raise TypeError(
                        f"{_DECORATOR_NAMES[kind]} 注入函数 {fname} 的上下文"
                        f"参数 {name!r} 不得有默认值——上下文是框架保留名，"
                        "必然由框架按名填充，默认值永远不会生效")
                context.append(name)
        missing = sorted(required - set(context))
        if missing:
            raise TypeError(
                f"{_DECORATOR_NAMES[kind]} 注入函数 {fname} 缺少必选上下文"
                f"参数 {missing}——注入纪律要求显式接收 "
                f"{sorted(required)}（全部由框架在 apply 时按名填充，用户只管使用"
                "——用户只管使用）")
        fn._injection_meta = InjectionMeta(kind=kind, context=frozenset(context))
        return fn

    return decorator


local_compute = _make_decorator(LOCAL_COMPUTE)
inner_wrapper = _make_decorator(INNER_WRAPPER)


def require_injection_meta(fn, kind, *, source):
    """取注入函数的元数据；未装饰 / 种类不符 → fail-fast（教学式报错）。"""
    meta = getattr(fn, "_injection_meta", None)
    name = getattr(fn, "__name__", fn)
    if meta is None:
        raise TypeError(
            f"{source}: 注入函数 {name} 缺少 {_DECORATOR_NAMES[kind]} 装饰器"
            "——显式注入纪律要求所有注入函数带模板装饰器（装饰器声明所需的"
            "框架上下文并保证签名可校验）：local_compute_fn 的运行时 fn 用 "
            "@local_compute（区域计算工厂），inner_wrapper 用 "
            "@inner_wrapper（hyper_models.components.distributed 导出）")
    if meta.kind != kind:
        raise TypeError(
            f"{source}: 注入函数 {name} 的装饰器种类不匹配——got "
            f"{_DECORATOR_NAMES[meta.kind]}，期望 {_DECORATOR_NAMES[kind]}")
    return meta


def fill_context_kwargs(meta, context, configured=None, *, source=""):
    """按 meta 声明的上下文键取框架填充的 kwargs（极简、无隐藏行为）。

    - 填充集合 == 声明集合，一个不多一个不少（每次填充记 INFO）；
    - 上下文键是框架保留名：用户在 Target/YAML 里配置了同名键 →
      fail-fast（上下文参数无默认值，YAML resolver 不会回填它们，因此
      出现在 configured 里的保留键一定是用户显式写的）。
    """
    configured = configured or {}
    reserved = sorted(set(configured) & meta.context)
    if reserved:
        raise ValueError(
            f"{source}: 配置了框架保留上下文键 {reserved}——上下文由框架按"
            "声明填充，不可配置；你的配置键不能与保留名同名")
    kwargs = {}
    for key in sorted(meta.context):
        kwargs[key] = context[key]
        logger.info("%s: 上下文键 %s 由框架填充 (%s)",
                    source, key, type(context[key]).__name__)
    return kwargs


# ────────────────────────────────────────────────────────────────────────────
# 运行时层签名校验（原则 1：注入函数的入参必须与原函数匹配）
# ────────────────────────────────────────────────────────────────────────────

def _is_subsequence(sub, full) -> bool:
    it = iter(full)
    return all(name in it for name in sub)


def validate_local_compute_signature(compute_fn, forward, *, owner):
    """校验区域 compute fn 的入参与原 forward 匹配（apply 时 fail-fast）。

    规则（compute_fn 首参是 module，forward 首参 self，均跳过）：
    1. compute fn 不得有 *args/**kwargs（骨架按 forward 实参透传，吞参会
       掩盖不匹配）；
    2. compute fn 的每个入参必须在 forward 中有**同名**形参（骨架会以
       kwarg 透传，名字对不上即运行期 TypeError）；
    3. compute fn 的位置参数序列必须是 forward 位置参数序列的**子序列**
       （位置透传不得乱序）；
    4. forward 的全部必填参数（无默认值）compute fn 必须接得住。
    """
    try:
        fn_params = list(inspect.signature(compute_fn).parameters.values())
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{owner}: 无法内省注入 compute fn 的签名") from exc
    if not fn_params:
        raise TypeError(
            f"{owner}: compute fn 至少需要 module 首参——契约是 "
            "fn(module, *forward_args)")
    fn_params = fn_params[1:]                     # 跳过 module
    for p in fn_params:
        if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                      inspect.Parameter.VAR_KEYWORD):
            raise TypeError(
                f"{owner}: compute fn 不允许 *args/**kwargs（形参 {p.name!r}）"
                "——入参必须与原 forward 显式匹配")

    fwd_params = list(inspect.signature(forward).parameters.values())
    fwd_names = {p.name for p in fwd_params
                 if p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                   inspect.Parameter.VAR_KEYWORD)}
    fn_names = {p.name for p in fn_params}
    for p in fn_params:
        if p.name not in fwd_names:
            raise TypeError(
                f"{owner}: compute fn 的入参 {p.name!r} 在原 forward 的形参 "
                f"{sorted(fwd_names)} 中不存在同名项——注入函数的入参必须与"
                "原函数匹配（骨架按 forward 实参透传，名字对不上会在运行期"
                " TypeError）")
    fwd_pos = [p.name for p in fwd_params
               if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                             inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    fn_pos = [p.name for p in fn_params
              if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    if not _is_subsequence(fn_pos, fwd_pos):
        raise TypeError(
            f"{owner}: compute fn 的位置参数 {fn_pos} 不是原 forward 位置"
            f"参数 {fwd_pos} 的同序子序列——骨架按位置透传实参，乱序会接错")
    required = [p.name for p in fwd_params
                if p.default is inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                   inspect.Parameter.VAR_KEYWORD)]
    missing = [n for n in required if n not in fn_names]
    if missing:
        raise TypeError(
            f"{owner}: 原 forward 的必填参数 {missing} 未被 compute fn 接收"
            "（compute fn 声明的入参必须覆盖原函数的必填项）")


def validate_wrapped_forward(orig_forward, new_forward, *, owner):
    """校验 inner_wrapper 替换后的 forward 能接收原 forward 的全部入参。

    用原签名构造一次 dummy bind 试探（*args/**kwargs 参数无法伪造，跳
    过）；替换侧允许 *args/**kwargs 宽容透传（它必须把不关心的实参传给
    原实现），但原 forward 的具名参数必须全部可按名绑定。
    """
    orig_sig = inspect.signature(orig_forward)
    try:
        new_sig = inspect.signature(new_forward, follow_wrapped=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{owner}: 无法内省注入 wrapper 替换后的 forward 签名") from exc
    args, kwargs = [], {}
    for p in orig_sig.parameters.values():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                      inspect.Parameter.VAR_KEYWORD):
            continue
        if p.kind is inspect.Parameter.POSITIONAL_ONLY:
            args.append(None)
        else:
            # POSITIONAL_OR_KEYWORD 按名（kwarg）试探——名称维度的契约才是
            # 注入纪律关心的；可选位置实参的纯位置传法不做要求
            kwargs[p.name] = None
    try:
        new_sig.bind(*args, **kwargs)
    except TypeError as exc:
        raise TypeError(
            f"{owner}: 注入 wrapper 替换后的 forward 与原 forward 入参不兼容"
            f"（{exc}）——原签名 {orig_sig}，替换后签名 {new_sig}；替换后的 "
            "forward 必须能接收原 forward 的全部入参（可用 *args/**kwargs "
            "宽容透传）") from exc
