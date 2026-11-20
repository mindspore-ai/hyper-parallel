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
"""Stock-Transformers fixes for the Qwen3-Omni family.

The installed stock ``transformers`` (e.g. 4.57.1) ships several defects that
break Qwen3-Omni training on NPU: a packed-audio dead branch, a Conv3d
backward segfault, a deepstack 3-D-mask joint-branch crash, and
generation-only wrappers without a training ``forward``. Upstream families
fix stock defects through codegen ``generated/`` bundles; until the family
graduates to a generated modeling file, these idempotent runtime patches
carry the same fixes. All patches guard on the buggy stock pattern so an
already-fixed modeling is left untouched.
"""

import inspect
import logging
import textwrap

logger = logging.getLogger(__name__)

_QWEN3_OMNI_MOE_MODELING = "transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe"
_QWEN2_5_OMNI_MODELING = "transformers.models.qwen2_5_omni.modeling_qwen2_5_omni"


def _import_class(module_name, class_name):
    """Return the class from ``module_name`` or ``None`` when unavailable."""
    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError):
        return None


def ensure_thinker_forward() -> None:
    """Give generation-only omni wrappers a training ``forward`` -> thinker.

    Idempotent and defensive: only classes that exist in the installed
    Transformers and that do not already define their own ``forward`` are
    patched. The delegating forward filters keyword arguments to those the
    thinker's ``forward`` accepts, which drops shared-VLM-pipeline fields the
    omni thinker does not consume (e.g. ``mm_token_type_ids`` — omni derives
    multimodal RoPE from ``input_ids`` directly).
    """
    candidates = []
    for module_name, class_name in (
        (_QWEN3_OMNI_MOE_MODELING, "Qwen3OmniMoeForConditionalGeneration"),
        (_QWEN2_5_OMNI_MODELING, "Qwen2_5OmniForConditionalGeneration"),
    ):
        model_cls = _import_class(module_name, class_name)
        if model_cls is not None:
            candidates.append((model_cls, class_name))

    for model_cls, class_name in candidates:
        # Declare the nested decoder-layer containers for distributed.compile
        # regardless of whether the training-forward patch applies below.
        if "get_compile_layers" not in model_cls.__dict__:
            model_cls.get_compile_layers = _get_compile_layers
        if "forward" in model_cls.__dict__:
            continue  # respects an upstream training forward if one is added later
        # The thinker attribute is instance-level; inspect the thinker *class*
        # forward to learn which kwargs to forward.
        thinker_cls_name = (
            "Qwen3OmniMoeThinkerForConditionalGeneration"
            if class_name.startswith("Qwen3")
            else "Qwen2_5OmniThinkerForConditionalGeneration"
        )
        module_name = (
            _QWEN3_OMNI_MOE_MODELING if class_name.startswith("Qwen3") else _QWEN2_5_OMNI_MODELING
        )
        thinker_cls_forward = None
        thinker_cls = _import_class(module_name, thinker_cls_name)
        if thinker_cls is not None:
            thinker_cls_forward = thinker_cls.forward
        if thinker_cls_forward is None:
            continue
        params = inspect.signature(thinker_cls_forward).parameters
        accepts_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        param_names = {name for name in params if name != "self"}
        model_cls.forward = _make_thinker_forward(param_names, accepts_var_kw)
        logger.info("Patched %s.forward to delegate to thinker (training path).", class_name)


def _get_compile_layers(self):
    """Declare decoder-layer containers for ``distributed.compile``.

    Qwen3-Omni nests its MoE decoder under ``thinker.model.layers`` (and the
    talker under ``talker.model.layers`` when audio output is enabled), which
    the generic HF-convention paths (``layers``, ``model.layers``) cannot
    reach. This satisfies the model-owned ``get_compile_layers()`` contract.
    """
    layers = []
    for path in ("thinker.model.layers", "talker.model.layers"):
        container = self
        for part in path.split("."):
            container = getattr(container, part, None)
            if container is None:
                break
        if container is not None:
            layers.extend(container)
    return layers


def _make_thinker_forward(param_names, accepts_var_kw):
    """Build a wrapper ``forward`` delegating to ``self.thinker``."""

    def thinker_forward(self, **kwargs):
        if not accepts_var_kw:
            kwargs = {k: v for k, v in kwargs.items() if k in param_names}
        else:
            # Thinker takes **kwargs; still drop fields the omni stack never
            # consumes so they do not propagate into text/vision submodules.
            kwargs.pop("mm_token_type_ids", None)
        # The VLM collator produces packed (T_total, n_mels) input_features
        # + per-audio audio_feature_lengths, but the stock thinker forward
        # nulls audio_feature_lengths whenever
        # feature_attention_mask is None — get_rope_index then crashes on
        # audio_seqlens[...]. Rebuild the batched padded layout here so the
        # forward computes audio_feature_lengths itself; the patched
        # get_audio_features re-packs for the tower (same sample order).
        packed = kwargs.get("input_features")
        lens = kwargs.get("audio_feature_lengths")
        if (
            packed is not None
            and lens is not None
            and kwargs.get("feature_attention_mask") is None
            and packed.dim() == 2
        ):
            import torch  # pylint: disable=import-outside-toplevel

            lens = torch.as_tensor(lens)
            n_mels = packed.shape[-1]
            tmax = int(lens.max())
            batched = packed.new_zeros((lens.numel(), n_mels, tmax))
            mask = packed.new_zeros((lens.numel(), tmax), dtype=torch.long)
            ofs = 0
            for i, length in enumerate(lens.tolist()):
                batched[i, :, :length] = packed[ofs : ofs + length].t()
                mask[i, :length] = 1
                ofs += length
            kwargs["input_features"] = batched
            kwargs["feature_attention_mask"] = mask
            kwargs.pop("audio_feature_lengths", None)
        return self.thinker(**kwargs)

    return thinker_forward


def ensure_packed_audio_input() -> None:
    """Make packed audio input reach the omni audio tower.

    The audio tower consumes packed ``(T_total, 128)`` mel features plus
    per-audio ``feature_lens`` — the layout the data pipeline emits. Stock
    4.57.1 ``get_audio_features`` discards the caller's
    ``audio_feature_lengths`` whenever ``feature_attention_mask`` is None,
    so the packed layout can never reach the tower (and both being None
    crashes with AttributeError). The replacement accepts both layouts:
    batched-padded ``(B, 128, T)`` (select valid frames, transpose) and
    packed ``(T_total, 128)`` (transpose only), and casts to the model dtype
    before the convs (fp32 mel vs bf16 conv weights segfaults on NPU).
    Guarded on the buggy stock pattern so an already-fixed modeling is left
    untouched; idempotent.
    """
    import torch  # pylint: disable=import-outside-toplevel

    # The buggy stock pattern: the else branch nulls the caller's
    # audio_feature_lengths whenever feature_attention_mask is None.
    buggy = (
        "        if feature_attention_mask is not None:\n"
        "            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)\n"
        "        else:\n"
        "            audio_feature_lengths = None\n"
    )
    targets = []
    for module_name, class_name in (
        (_QWEN3_OMNI_MOE_MODELING, "Qwen3OmniMoeThinkerForConditionalGeneration"),
        (_QWEN2_5_OMNI_MODELING, "Qwen2_5OmniThinkerForConditionalGeneration"),
    ):
        thinker_cls = _import_class(module_name, class_name)
        if thinker_cls is not None:
            targets.append(thinker_cls)
    for thinker_cls in targets:
        if getattr(thinker_cls, "_hyper_parallel_packed_audio", False):
            continue
        # Only replace the broken stock body; a fixed upstream modeling that
        # already forwards audio_feature_lengths is left untouched.
        if buggy not in inspect.getsource(thinker_cls.get_audio_features):
            logger.warning(
                "%s.get_audio_features: packed-audio nulling pattern not found — skipping fix (non-fatal).",
                thinker_cls.__name__,
            )
            continue

        def _get_audio_features(
            self,
            input_features,
            feature_attention_mask=None,
            audio_feature_lengths=None,
        ):
            if feature_attention_mask is not None:
                # Batched padded layout (B, 128, T): select valid frames and
                # transpose to the tower's channel-first (128, T_total).
                audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
                input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
            elif input_features.dim() == 2 and input_features.shape[-1] == 128:
                # Packed layout (T_total, 128) from the VLM collator: transpose
                # to the tower's channel-first (128, T_total).
                input_features = input_features.t()
            if audio_feature_lengths is None:
                raise ValueError(
                    "packed audio requires audio_feature_lengths (or feature_attention_mask)"
                )
            # fp32 mel features vs bf16 conv weights segfault the kernel on
            # NPU, so cast first. qwen3_omni_moe's tower exposes ``conv2d1``;
            # qwen2_5_omni's uses ``conv1`` (1-D), so only the qwen3_omni_moe
            # target reaches this cast with audio flowing.
            input_features = input_features.to(self.audio_tower.conv2d1.weight.dtype)
            audio_outputs = self.audio_tower(input_features, feature_lens=audio_feature_lengths)
            return audio_outputs.last_hidden_state

        thinker_cls.get_audio_features = _get_audio_features
        thinker_cls._hyper_parallel_packed_audio = True
        logger.info(
            "Patched %s.get_audio_features -> packed (T_total,128)+feature_lens.",
            thinker_cls.__name__,
        )


def ensure_patch_embed_linear() -> None:
    """Route the omni vision ``patch_embed`` Conv3d through an equivalent Linear.

    CANN 8.5 (torch_npu 2.6) segfaults in ``aclnnConvolutionBackwardGetWorkspaceSize``
    for exactly the Qwen3-Omni patch-embed Conv3d configuration
    (in=3, out=1152, kernel/stride=(2,16,16), input (N,3,2,16,16), bf16) — a
    minimal Conv3d backward segfaults in libopapi_nn.so. Because the patch-embed
    kernel exactly covers the whole receptive field (kernel == input spatial/\
    temporal extent), the Conv3d is numerically identical to a Linear over the
    flattened feature map. Routing it through ``F.linear`` with a view of the
    conv weight: (a) produces bit-identical forward output, (b) backprops a
    gradient into the very same ``conv.weight`` storage via the contiguous view,
    and (c) avoids the broken conv backward path entirely. Applied only to the
    omni vision patch embed; idempotent.
    """
    module = _import_class(_QWEN3_OMNI_MOE_MODELING, "modeling_qwen3_omni_moe")
    if module is None:
        return
    name = "Qwen3OmniMoeVisionPatchEmbed"
    patch_embed_cls = getattr(module, name, None)
    if patch_embed_cls is None:
        return
    if getattr(patch_embed_cls, "_hyper_parallel_linear_patch", False):
        return
    try:
        from torch.nn import functional as _F  # pylint: disable=import-outside-toplevel
    except ImportError:  # pragma: no cover
        return

    original_forward = patch_embed_cls.forward

    def linear_forward(self, hidden_states):
        # Same reshape the original forward performs before the Conv3d.
        x = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        kernel = self.proj.kernel_size
        stride = self.proj.stride
        if tuple(stride) == tuple(kernel) and tuple(x.shape[-3:]) == tuple(kernel):
            B = x.shape[0]
            w = self.proj.weight.view(self.proj.weight.shape[0], -1)
            return _F.linear(x.view(B, -1), w, self.proj.bias).view(B, -1)
        # Fall back to the stock Conv3d for any other receptive field.
        return original_forward(self, hidden_states)

    patch_embed_cls.forward = linear_forward
    patch_embed_cls._hyper_parallel_linear_patch = True
    logger.info("Patched %s.forward -> equivalent Linear (CANN conv3d-backward workaround).", name)


def ensure_deepstack_2d_masks() -> None:
    """Fix the image+video deepstack joint-branch mask-shape crash.

    Before the deepstack scatter, ``forward`` expands the placeholder masks
    to 3-D via ``get_placeholder_mask``. The image+video joint branch then
    boolean-indexes with those 3-D masks, producing ``S*H`` rows where the
    per-modality feature tensors have ``S`` rows — ``RuntimeError: shape
    mismatch``. The convention downstream (``_deepstack_process`` squeezes
    with ``[..., 0]``) is 2-D token-dim masks. The fix squeezes the joint
    masks to the token dim inside the joint branch and makes the downstream
    squeeze conditional. Applied as a targeted source rewrite of the
    thinker ``forward`` (guarded on the buggy pattern being present and not
    already patched); idempotent.
    """
    module = _import_class(_QWEN3_OMNI_MOE_MODELING, "modeling_qwen3_omni_moe")
    if module is None:
        return
    thinker_cls = getattr(module, "Qwen3OmniMoeThinkerForConditionalGeneration", None)
    text_model_cls = getattr(module, "Qwen3OmniMoeThinkerTextModel", None)
    if thinker_cls is None:
        return
    if getattr(thinker_cls, "_hyper_parallel_deepstack_2d_masks", False):
        return

    # --- fix 1: joint branch inside thinker.forward (source rewrite) ---
    source = inspect.getsource(thinker_cls.forward)
    # ``inspect.getsource`` includes BOTH decorators (``@can_return_tuple``
    # and ``@auto_docstring``); dedent then re-exec would re-run them on a
    # bare function and crash (auto_class_docstring expects a class) — strip
    # down to the ``def``. Training always runs with the default
    # ``return_dict=True``, so dropping ``can_return_tuple`` is safe here.
    source = source[source.index("def forward") :]
    buggy = (
        "visual_pos_masks = video_mask | image_mask\n"
        "                visual_embeds_multiscale_joint = ()\n"
        "                image_mask_joint = image_mask[visual_pos_masks]\n"
        "                video_mask_joint = video_mask[visual_pos_masks]"
    )
    fixed = (
        "visual_pos_masks = (video_mask | image_mask)[..., 0]\n"
        "                # masks are 3-D (expand_as) here; squeeze to token dim\n"
        "                visual_embeds_multiscale_joint = ()\n"
        "                image_mask_joint = image_mask[..., 0][visual_pos_masks]\n"
        "                video_mask_joint = video_mask[..., 0][visual_pos_masks]"
    )
    if buggy not in source:
        logger.warning(
            "deepstack joint-branch pattern not found — skipping fix (non-fatal)."
        )
        return
    source = source.replace(buggy, fixed)
    namespace = dict(module.__dict__)
    exec(compile(textwrap.dedent(source), "<omni_deepstack_fix>", "exec"), namespace)  # noqa: S102
    thinker_cls.forward = namespace["forward"]
    thinker_cls._hyper_parallel_deepstack_2d_masks = True

    # --- fix 2: downstream squeeze must tolerate 2-D masks ---
    if text_model_cls is not None and not getattr(text_model_cls, "_hyper_parallel_deepstack_2d", False):

        def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
            if visual_pos_masks.ndim == 3:
                visual_pos_masks = visual_pos_masks[..., 0]
            visual_pos_masks = visual_pos_masks.to(hidden_states.device)
            local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
            hidden_states[visual_pos_masks, :] = local_this
            return hidden_states

        text_model_cls._deepstack_process = _deepstack_process
        text_model_cls._hyper_parallel_deepstack_2d = True

    logger.info(
        "Patched %s.forward joint deepstack branch + %s._deepstack_process "
        "(3-D mask index crash fix).",
        thinker_cls.__name__,
        text_model_cls.__name__ if text_model_cls else "-",
    )


def ensure_moe_fused_npu() -> None:
    """Patch the omni sparse-MoE to NPU fused kernels when opted in.

    Optional performance path gated on ``HYPER_OMNI_MOE_FUSED_NPU=1``; the
    stock expert loop runs otherwise. Implementation lives in the family
    adapter module :mod:`.omni_moe_fused_npu`.
    """
    from hyper_parallel.models.qwen3_omni.adapter.omni_moe_fused_npu import (  # pylint: disable=import-outside-toplevel
        _ensure_omni_moe_fused_npu as _ensure_impl,
    )

    _ensure_impl()


def apply_stock_model_patches() -> None:
    """Apply every stock-modeling fix for the Qwen3-Omni family (idempotent).

    Called by :meth:`HyperAutoModelForMultimodalLM.from_pretrained` and by
    the family registration's ``materialization`` provider before the model
    is built, so both the auto-model and the planner-driven paths see a
    fixed modeling.
    """
    ensure_thinker_forward()
    # HF 4.57.1 stock-bug fixes (details in each ensure_* docstring):
    # packed-audio dead branch / Conv3d-backward NPU segfault / deepstack
    # 3-D-mask joint-branch crash.
    ensure_packed_audio_input()
    ensure_patch_embed_linear()
    ensure_deepstack_2d_masks()
    # Optional: fused NPU MoE kernels, opt-in via HYPER_OMNI_MOE_FUSED_NPU=1;
    # stock forward otherwise.
    ensure_moe_fused_npu()


__all__ = [
    "apply_stock_model_patches",
    "ensure_deepstack_2d_masks",
    "ensure_moe_fused_npu",
    "ensure_packed_audio_input",
    "ensure_patch_embed_linear",
    "ensure_thinker_forward",
]
