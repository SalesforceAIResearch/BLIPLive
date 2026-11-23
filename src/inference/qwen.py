"""Qwen3.5 base-model inference, extracted from stream-gemma."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass
class QwenRuntime:
    model: object
    processor: object
    device: object

    @classmethod
    def load(cls, *, model_name, model_revision):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("The released WEB configurations require a CUDA GPU")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        processor = AutoProcessor.from_pretrained(model_name, revision=model_revision)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            revision=model_revision,
            use_safetensors=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": local_rank},
        )
        return cls(model.eval(), processor, torch.device(f"cuda:{local_rank}"))

    def _prepare_inputs(self, messages: list[dict]) -> dict:
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    @torch.no_grad()
    def generate_messages(self, messages: list[dict], *, max_new_tokens: int) -> str:
        inputs = self._prepare_inputs(messages)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        input_len = int(inputs["input_ids"].shape[1])
        return self.processor.batch_decode(
            output[:, input_len:],
            skip_special_tokens=True,
        )[0].strip()
