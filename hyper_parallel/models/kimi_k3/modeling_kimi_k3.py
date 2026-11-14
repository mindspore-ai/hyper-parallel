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
"""Complete K3 image/text training model using pre-expanded media coordinates."""

from __future__ import annotations

from typing import Any

# Native Torch model, consistent with the DeepSeek-V4.1 model integration.
import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from transformers import PreTrainedModel

from hyper_parallel.models.kimi_k3.configuration_kimi_k3 import KimiK3Config
from hyper_parallel.models.kimi_k3.modeling_kimi_linear import KimiLinearForCausalLM
from hyper_parallel.models.kimi_k3.vision import (
    InterpolatedPositionEmbedding,
    MoonViT3dPretrainedModel,
    PatchMergerMLPV2,
)


class KimiK3ForConditionalGeneration(PreTrainedModel):
    """HF-compatible K3 architecture with explicit pretraining media semantics."""

    config_class = KimiK3Config
    base_model_prefix = "language_model"
    _no_split_modules = ["KimiDecoderLayer", "MoonViTEncoderLayer", "PatchMergerMLPV2"]
    _supports_sdpa = True
    _supports_flash_attn_2 = False

    def __init__(self, config: KimiK3Config) -> None:
        """Construct every branch without requiring recipe replacements."""
        if not isinstance(config, KimiK3Config):
            config = KimiK3Config(**config.to_dict())
        if (getattr(config, "quantization_config", None)
                or getattr(config.text_config, "quantization_config", None)):
            raise ValueError("K3 scratch training requires a floating-point config without quantization_config")
        if config.vision_config.text_hidden_size != config.text_config.hidden_size:
            raise ValueError("vision_config.text_hidden_size must match text_config.hidden_size")
        if config.ignore_index != -100:
            raise ValueError("K3 training uses the framework ignore_index=-100")
        super().__init__(config)
        self.vision_tower = MoonViT3dPretrainedModel(config.vision_config)
        self.mm_projector = PatchMergerMLPV2(config.vision_config)
        self.language_model = KimiLinearForCausalLM(config.text_config)
        self.vocab_size = config.text_config.vocab_size
        self.post_init()

    @classmethod
    def from_config(cls, config: KimiK3Config, **kwargs: Any) -> "KimiK3ForConditionalGeneration":
        """Construct the model through the HyperAutoModel custom-model protocol.

        Args:
            config: Native image/text architecture configuration.
            **kwargs: Unsupported; construction options belong in config.

        Returns:
            A newly initialized image/text model.

        Raises:
            ValueError: Extra construction options are supplied.
        """
        if kwargs:
            raise ValueError(f"Unsupported K3 construction options: {sorted(kwargs)}")
        return cls(config)

    def _init_weights(self, module: nn.Module) -> None:
        self.language_model._init_weights(module)  # pylint: disable=protected-access
        if isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=self.config.text_config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, InterpolatedPositionEmbedding):
            module.reset_parameters()

    def get_input_embeddings(self) -> nn.Embedding:
        """Expose the text embeddings through the standard HF contract."""
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Replace the language model's input embedding module.

        Args:
            value: Replacement module with the configured vocabulary and hidden dimensions.
        """
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Linear:
        """Expose the text vocabulary projection."""
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, value: nn.Linear) -> None:
        """Replace the language model's vocabulary projection module.

        Args:
            value: Replacement module with the configured vocabulary and hidden dimensions.
        """
        self.language_model.set_output_embeddings(value)

    def _vision_features(self, pixel_values: torch.Tensor | None, grid_thws: torch.Tensor | None,
                         embeddings: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Encode images while preserving collective order for text-only samples."""
        if pixel_values is None or pixel_values.numel() == 0:
            if grid_thws is not None and grid_thws.numel():
                raise ValueError("grid_thws cannot describe media when pixel_values is absent")
            # Every rank visits the same FSDP units even for a text-only microbatch.
            patch_size = self.config.vision_config.patch_size
            pixels = embeddings.new_zeros((4, 3, patch_size, patch_size))
            grids = torch.tensor([[1, 2, 2]], dtype=torch.long, device=embeddings.device)
            features = self.mm_projector(self.vision_tower(pixels, grids))
            return torch.cat(features), False
        if grid_thws is None or grid_thws.shape[0] == 0:
            raise ValueError("nonempty pixel_values require grid_thws")
        features = self.mm_projector(self.vision_tower(pixel_values, grid_thws))
        return torch.cat(features), True

    def _merge_media(self, input_ids: torch.Tensor, embeddings: torch.Tensor, features: torch.Tensor,
                     grid_thws: torch.Tensor, media_spans: torch.Tensor | None,
                     media_token_mask: torch.Tensor | None,
                     media_patch_offsets: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Place each encoded image into its pre-expanded token span."""
        if media_spans is None or media_spans.shape != (grid_thws.shape[0], 3):
            raise ValueError("pre-expanded K3 images require media_spans [media_count,3] (batch,start,end)")
        if media_spans.dtype not in (torch.int32, torch.int64):
            raise ValueError("media_spans must have an integer dtype")
        expected_mask = self._media_span_mask(input_ids, grid_thws, media_spans)
        placeholder_mask = input_ids.eq(self.config.media_placeholder_token_id)
        if not torch.equal(expected_mask, placeholder_mask):
            raise ValueError("expanded placeholder positions must exactly match media_spans")
        if media_token_mask is not None and (
                media_token_mask.shape != input_ids.shape or not torch.equal(media_token_mask.bool(), expected_mask)):
            raise ValueError("media_token_mask must exactly match expanded media spans")
        if media_patch_offsets is not None:
            lengths = grid_thws.prod(-1)
            expected_offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
            if not torch.equal(media_patch_offsets.to(expected_offsets.device), expected_offsets):
                raise ValueError("media_patch_offsets must partition the grid patch sequence")
        indices = expected_mask.flatten().nonzero().flatten()
        if features.shape[0] != indices.numel():
            raise ValueError("visual features and expanded placeholder counts differ")
        merged = embeddings.flatten(0, 1).index_copy(0, indices, features.to(embeddings.dtype))
        return merged.reshape(embeddings.shape), expected_mask

    @staticmethod
    def _media_span_mask(input_ids: torch.Tensor, grid_thws: torch.Tensor,
                         media_spans: torch.Tensor) -> torch.Tensor:
        """Resolve ordered media spans to decoder-token ownership."""
        grids = grid_thws.tolist()
        spans = media_spans.tolist()
        expected_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        previous_end = -1
        batch_size, sequence_length = input_ids.shape
        for (batch_index, start, end), (_, height, width) in zip(spans, grids):
            if not 0 <= batch_index < batch_size or not 0 <= start < end <= sequence_length:
                raise ValueError("media_spans must lie inside the expanded batch sequence")
            if end - start != height * width // 4:
                raise ValueError("media span length must equal the post-merge h*w/4 feature count")
            if batch_index * sequence_length + start < previous_end:
                raise ValueError("media_spans must be ordered by sample/position without overlap")
            previous_end = batch_index * sequence_length + end
            expected_mask[batch_index, start:end] = True
        return expected_mask

    def _parallel_embeddings(self, input_ids: torch.Tensor, embeddings: torch.Tensor,
                             pixel_values: torch.Tensor, grid_thws: torch.Tensor,
                             media_spans: torch.Tensor, media_token_mask: torch.Tensor | None,
                             media_patch_offsets: torch.Tensor, plan: Any, encoder_parallel_inputs: Any = None
                             ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge only feature rows owned by this decoder interval."""
        plan.decoder_layout.validate_hidden(embeddings)
        if not plan.active:
            if bool(input_ids.eq(self.config.media_placeholder_token_id).any()) or (
                    media_token_mask is not None and bool(media_token_mask.any())):
                raise ValueError("text-only encoder plans cannot contain media placeholders")
            features, _ = self._vision_features(pixel_values, grid_thws, embeddings)
            return embeddings + features.sum().to(embeddings.dtype) * 0.0, torch.zeros_like(input_ids, dtype=torch.bool)
        image_inputs = encoder_parallel_inputs if encoder_parallel_inputs is not None else {
            "pixel_values": pixel_values, "grid_thws": grid_thws,
            "media_spans": media_spans, "media_patch_offsets": media_patch_offsets,
        }
        records, anchor = plan.encode_and_route(image_inputs)
        local_start = plan.decoder_layout.global_start
        local_stop = plan.decoder_layout.global_stop
        expected_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        merged = embeddings.clone()
        for item, features in records:
            first = max(item.decoder_token_start, local_start)
            last = min(item.decoder_token_start + item.output_rows, local_stop)
            if first >= last:
                raise ValueError("routed media does not intersect the decoder owner")
            feature_start = first - item.decoder_token_start
            merged[item.batch_index, first - local_start:last - local_start] = features[
                feature_start:feature_start + last - first].to(embeddings.dtype)
            expected_mask[item.batch_index, first - local_start:last - local_start] = True
        if not torch.equal(expected_mask, input_ids.eq(self.config.media_placeholder_token_id)):
            raise ValueError("local placeholders must exactly match routed media features")
        if media_token_mask is not None:
            if media_token_mask.shape[1] == plan.decoder_layout.global_sequence_length:
                media_token_mask = media_token_mask[:, local_start:local_stop]
            if media_token_mask.shape != input_ids.shape or not torch.equal(media_token_mask.bool(), expected_mask):
                raise ValueError("local media mask differs from the feature ownership plan")
        if anchor is not None:
            merged = merged + anchor.to(merged.dtype) * 0.0
        return merged, expected_mask

    def prepare_embeddings(self, input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None,
                           pixel_values: torch.Tensor | None, grid_thws: torch.Tensor | None,
                           media_spans: torch.Tensor | None, media_token_mask: torch.Tensor | None,
                           media_patch_offsets: torch.Tensor | None, encoder_parallel_plan: Any = None,
                           encoder_parallel_inputs: Any = None
                           ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Fill pre-expanded media positions without changing sequence length.

        Args:
            input_ids: Token IDs [batch, sequence], exclusive with inputs_embeds.
                Required when nonempty media is present to validate placeholders.
            inputs_embeds: Precomputed [batch, sequence, hidden] text embeddings.
            pixel_values: Flattened image patches, or None for a text-only batch.
            grid_thws: Integer [media, 3] image grids matching pixel_values.
            media_spans: Ordered, disjoint [media, 3] rows (batch, start, end).
            media_token_mask: Optional boolean [batch, sequence] placeholder mask.
            media_patch_offsets: Optional cumulative patch counts [media + 1].
            encoder_parallel_plan: Shared encoder work and decoder feature ownership.
            encoder_parallel_inputs: Device patches already packed for this encoder owner.

        Returns:
            Embeddings with visual features inserted and the corresponding media
            mask, or None as the mask for text-only inputs. Text-only batches still
            visit the vision FSDP units to preserve collective order.

        Raises:
            ValueError: Token ownership, image metadata or placeholder positions disagree.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        embeddings = self.get_input_embeddings()(input_ids) if inputs_embeds is None else inputs_embeds
        if encoder_parallel_plan is not None:
            if input_ids is None:
                raise ValueError("encoder parallel requires input_ids for exact media ownership")
            return self._parallel_embeddings(input_ids, embeddings, pixel_values, grid_thws, media_spans,
                                             media_token_mask, media_patch_offsets, encoder_parallel_plan,
                                             encoder_parallel_inputs=encoder_parallel_inputs)
        if any(getattr(layer.self_attn, "_hp_cp_mesh", None) is not None
               for layer in self.language_model.model.layers):
            raise ValueError("K3 multimodal CP requires an enabled encoder_parallel plan")
        features, has_media = self._vision_features(pixel_values, grid_thws, embeddings)
        if has_media:
            if input_ids is None:
                raise ValueError("image fusion requires input_ids to validate expanded placeholders")
            return self._merge_media(input_ids, embeddings, features, grid_thws, media_spans,
                                     media_token_mask, media_patch_offsets)
        if input_ids is not None and bool(input_ids.eq(self.config.media_placeholder_token_id).any()):
            raise ValueError("media placeholders require corresponding pixels and metadata")
        if media_spans is not None and media_spans.numel():
            raise ValueError("text-only inputs cannot carry nonempty media_spans")
        if media_token_mask is not None and bool(media_token_mask.any()):
            raise ValueError("text-only inputs cannot mark media tokens")
        return embeddings + features.sum().to(embeddings.dtype) * 0, None

    def forward(self, input_ids: torch.Tensor | None = None, inputs_embeds: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None,
                pixel_values: torch.Tensor | None = None, grid_thws: torch.Tensor | None = None,
                media_spans: torch.Tensor | None = None, media_token_mask: torch.Tensor | None = None,
                media_patch_offsets: torch.Tensor | None = None, media_metadata: Any = None,
                encoder_parallel_plan: Any = None, encoder_parallel_inputs: Any = None,
                labels_are_shifted: bool = False,
                shift_labels: torch.Tensor | None = None,
                chunk_loss_targets: torch.Tensor | None = None, chunk_loss_mask: torch.Tensor | None = None,
                chunk_loss_chunk_size: int = 1024, chunk_loss_ignore_index: int = -100,
                **kwargs: Any) -> Any:
        """Train image/text batches with raw labels or aligned chunk targets.

        Args:
            input_ids: Expanded token IDs [batch, sequence], exclusive with inputs_embeds.
            inputs_embeds: Precomputed text embeddings, supported for text-only inputs.
            attention_mask: Optional [batch, sequence] valid-token mask.
            labels: Raw unshifted token labels, with -100 for ignored positions.
                Exclusive with chunk_loss_targets; visual token labels are ignored.
            pixel_values: Flattened image patches, or None for text-only inputs.
            grid_thws: Integer [media, 3] image grids matching pixel_values.
            media_spans: Ordered [media, 3] rows (batch, start, end) in the expanded sequence.
            media_token_mask: Optional boolean mask matching expanded visual positions.
            media_patch_offsets: Optional cumulative counts partitioning image patches.
            media_metadata: Optional data provenance retained for input auditing; it
                does not change embeddings, positions or the language-model objective.
            encoder_parallel_plan: Derived encoder execution and decoder ownership.
            encoder_parallel_inputs: Device patches already packed for this encoder owner.
            labels_are_shifted: Labels were globally shifted by the collator before CP.
            shift_labels: Explicit aligned labels, exclusive with raw labels.
            chunk_loss_targets: Already shifted next-token targets [batch, sequence].
            chunk_loss_mask: Optional mask selecting supervised chunk targets.
            chunk_loss_chunk_size: Number of hidden-state rows per vocabulary loss chunk.
            chunk_loss_ignore_index: Ignored target value; must remain -100.
            **kwargs: Additional native language-model forward options. Chunk loss
                requires return_dict=True.

        Returns:
            Native causal-LM outputs for the raw-label path, or a ChunkedCausalLMOutput
            containing loss_sum and valid_token_count without full vocabulary logits.

        Raises:
            ValueError: Inputs, label ownership, image metadata or chunk targets conflict.
        """
        del media_metadata
        if sum(value is not None for value in (labels, shift_labels, chunk_loss_targets)) > 1:
            raise ValueError("Specify only one of labels, shift_labels and chunk_loss_targets")
        embeddings, media_mask = self.prepare_embeddings(
            input_ids, inputs_embeds, pixel_values, grid_thws, media_spans,
            media_token_mask, media_patch_offsets, encoder_parallel_plan, encoder_parallel_inputs,
        )
        if labels is not None and media_mask is not None and not labels_are_shifted:
            labels = labels.masked_fill(media_mask, -100)
        return self.language_model(
            inputs_embeds=embeddings, attention_mask=attention_mask,
            labels=None if labels_are_shifted else labels,
            shift_labels=labels if labels_are_shifted else shift_labels,
            chunk_loss_targets=chunk_loss_targets, chunk_loss_mask=chunk_loss_mask,
            chunk_loss_chunk_size=chunk_loss_chunk_size, chunk_loss_ignore_index=chunk_loss_ignore_index,
            **kwargs,
        )
