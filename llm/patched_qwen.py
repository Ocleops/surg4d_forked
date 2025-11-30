"""
Patched Qwen2.5-VL classes that support passing custom patch features directly,
bypassing the visual encoder. Uses inheritance instead of monkey patching.
"""

import torch
from typing import Optional, Union
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VLModelOutputWithPast,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.utils import is_torchdynamo_compiling, TransformersKwargs
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack


class PatchedQwen2_5_VLModel(Qwen2_5_VLModel):
    """Qwen2_5_VLModel with support for custom patch features.
    
    This allows passing pre-computed vision features instead of raw pixel values,
    which is useful when features are computed elsewhere (e.g., stored in a scene graph).
    """

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
        """
        Forward pass with optional custom_patch_features support.
        
        Additional kwargs:
            custom_patch_features: Optional pre-computed vision features to use
                instead of computing them from pixel_values.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Get custom features from kwargs or from attribute (set externally for generation)
        custom_patch_features = kwargs.pop("custom_patch_features", None)
        if custom_patch_features is None and hasattr(self, "_custom_patch_features"):
            # Only use attribute during generation decoding steps (when no pixel_values)
            # The attribute is set/cleared externally by generate_with_vision_features
            if pixel_values is None:
                custom_patch_features = getattr(self, "_custom_patch_features")

        def _stack_features(features):
            if isinstance(features, (list, tuple)):
                return torch.cat(features, dim=0)
            return features

        if pixel_values is not None:
            if custom_patch_features is not None:
                image_embeds = custom_patch_features
            else:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = _stack_features(image_embeds).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (
                prefill_compiled_stage or prefill_noncompiled_stage
            ) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros(
                        (batch_size, seq_length), device=inputs_embeds.device
                    )
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids += delta.to(position_ids.device)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()


class PatchedQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """Qwen2_5_VLForConditionalGeneration with support for custom patch features.
    
    Overrides prepare_inputs_for_generation to properly forward custom_patch_features
    through the generation loop.
    """

    def __init__(self, config):
        super().__init__(config)
        # Swap inner model to use our patched version with custom_patch_features support
        self.model.__class__ = PatchedQwen2_5_VLModel

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        custom_patch_features=None,
        **kwargs,
    ):
        # During decoding steps, GenerationMixin provides only the last token in input_ids
        # but the full attention_mask. Align mask length to avoid shape mismatches
        # inside Qwen2_5_VL get_rope_index when indexing with the mask.
        if attention_mask is not None and input_ids is not None:
            # If lengths differ, slice mask to align with provided input_ids tokens
            if attention_mask.shape[-1] != input_ids.shape[-1]:
                attention_mask = attention_mask[..., -input_ids.shape[-1] :]
        
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )
        if custom_patch_features is not None:
            model_inputs["custom_patch_features"] = custom_patch_features
        return model_inputs

