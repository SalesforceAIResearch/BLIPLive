"""Gemma-4 direct-embedding inference; preserves the reference WEB runtime."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import Gemma4ForCausalLM, Gemma4ForConditionalGeneration

FRAME_RESOLUTION = 384


def square_resize(
    images: list[Image.Image], resolution: int = FRAME_RESOLUTION
) -> list[Image.Image]:
    """Match the training sampler's RGB 384x384 BICUBIC geometry."""
    target = (resolution, resolution)
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    return [image if image.size == target else image.resize(target, resampling) for image in images]


class GemmaTextModel(Gemma4ForCausalLM):
    # @use_experts_implementation: required marker for Transformers expert selection.
    @torch.no_grad()
    def encode_frames(self, pixel_values, image_position_ids):
        device = next(self.vision_tower.parameters()).device
        vision = self.vision_tower(
            pixel_values=pixel_values.to(device=device, dtype=self.dtype),
            pixel_position_ids=image_position_ids.to(device=device),
        ).last_hidden_state
        projected = self.embed_vision(inputs_embeds=vision).to(self.dtype)
        count = pixel_values.shape[0]
        if projected.shape[0] != count * self.config.frame_num_tokens:
            raise ValueError("Unexpected vision token count")
        return projected.view(count, self.config.frame_num_tokens, -1)


@dataclass
class GemmaRuntime:
    model: object
    tokenizer: object
    image_processor: object
    device: object
    frame_resolution: int = 384

    @classmethod
    def load(cls, *, model_name, model_revision):
        from transformers import AutoProcessor, AutoTokenizer
        from transformers.initialization import no_init_weights
        from transformers.integrations.moe import use_experts_implementation
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts

        if not torch.cuda.is_available():
            raise RuntimeError("The released WEB configurations require a CUDA GPU")
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        # Preserve the grouped expert kernel and weight transplant used by the reference.
        use_experts_implementation(
            Gemma4TextExperts, is_transposed=False, has_bias=False, has_gate=True
        )
        base = Gemma4ForConditionalGeneration.from_pretrained(
            model_name,
            revision=model_revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        values = base.config.text_config.to_dict()
        values.update(frame_num_tokens=256, _experts_implementation="grouped_mm")
        config = type(base.config.text_config)(**values)
        with no_init_weights():
            model = GemmaTextModel._from_config(config, attn_implementation="sdpa")
        model.model.load_state_dict(base.model.language_model.state_dict(), strict=True)
        model.lm_head.weight.data.copy_(base.lm_head.weight.data)
        if model.config.tie_word_embeddings:
            model.tie_weights()
        model.vision_tower = base.model.vision_tower.requires_grad_(False)
        model.embed_vision = base.model.embed_vision.requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=model_revision, use_fast=True, padding_side="left"
        )
        tokenizer.eos_token = "<turn|>"
        tokenizer.add_special_tokens({"additional_special_tokens": ["<v>"]})
        model.config.eos_token_id = tokenizer.eos_token_id
        processor = AutoProcessor.from_pretrained(
            model_name, revision=model_revision
        ).image_processor
        del base
        return cls(model.to(device).requires_grad_(False).eval(), tokenizer, processor, device)

    def encode_pils(self, images: list[Image.Image], *, chunk: int = 32) -> torch.Tensor:
        if not images:
            return torch.empty(0)
        images = square_resize(images, self.frame_resolution)
        processed = self.image_processor(images=images, return_tensors="pt")
        features = []
        for start in range(0, len(images), chunk):
            stop = min(start + chunk, len(images))
            features.append(
                self.model.encode_frames(
                    processed["pixel_values"][start:stop],
                    processed["image_position_ids"][start:stop],
                ).cpu()
            )
        return torch.cat(features)

    def embed_text(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
        return self.model.get_input_embeddings()(ids.to(self.device))

    def embed_token(self, token_id: int) -> torch.Tensor:
        ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        return self.model.get_input_embeddings()(ids)

    def image_marker_ids(self) -> tuple[int | None, int | None]:
        unknown = self.tokenizer.unk_token_id
        start = self.tokenizer.convert_tokens_to_ids("<start_of_image>")
        end = self.tokenizer.convert_tokens_to_ids("<end_of_image>")
        if start in (None, unknown) or end in (None, unknown):
            return None, None
        return int(start), int(end)

    def visual_block(self, frame: torch.Tensor) -> list[torch.Tensor]:
        start, end = self.image_marker_ids()
        tensors = []
        if start is not None:
            tensors.append(self.embed_token(start))
        tensors.append(frame.to(device=self.device, dtype=self.model.dtype).unsqueeze(0))
        if end is not None:
            tensors.append(self.embed_token(end))
        return tensors

    @torch.no_grad()
    def greedy_from_embeds(
        self,
        inputs: Iterable[torch.Tensor],
        *,
        max_new_tokens: int,
    ) -> str:
        block = torch.cat(list(inputs), dim=1)
        output = self.model(inputs_embeds=block, use_cache=True)
        past = output.past_key_values
        eos = int(self.model.config.eos_token_id)
        generated: list[int] = []
        next_id = int(output.logits[:, -1].argmax(dim=-1).item())
        for _ in range(max_new_tokens):
            if next_id == eos:
                break
            generated.append(next_id)
            output = self.model(
                inputs_embeds=self.embed_token(next_id),
                past_key_values=past,
                use_cache=True,
            )
            past = output.past_key_values
            next_id = int(output.logits[:, -1].argmax(dim=-1).item())
        return self.tokenizer.decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        ).strip()
