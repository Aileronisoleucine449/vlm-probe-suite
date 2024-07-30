"""LLaVA-1.5 adapter (frozen).

Uses transformers LlavaForConditionalGeneration with the image-text-matching
head. Slow; please use --max-samples during development.
"""
from __future__ import annotations
import torch
from .base import ModelAdapter

try:
    from transformers import AutoProcessor, LlavaForConditionalGeneration
except ImportError:  # pragma: no cover
    AutoProcessor = None
    LlavaForConditionalGeneration = None


class LlavaAdapter(ModelAdapter):
    def __init__(self, hf_id: str = "llava-hf/llava-1.5-7b-hf", device: str = "cuda"):
        super().__init__(device=device)
        if LlavaForConditionalGeneration is None:
            raise RuntimeError("transformers not installed")
        self.name = f"llava-{hf_id.split('/')[-1]}"
        self.model = LlavaForConditionalGeneration.from_pretrained(hf_id, torch_dtype=torch.float16).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(hf_id)

    @torch.no_grad()
    def encode_image(self, images):
        # llava embeds image *inside* the LM, so this surface is awkward
        # we expose vision tower features instead
        pixel = self.processor(images=list(images), return_tensors="pt").pixel_values.to(self.device).to(self.model.dtype)
        return self.model.vision_tower(pixel).last_hidden_state.mean(dim=1)

    @torch.no_grad()
    def encode_text(self, texts):
        # text-only branch: feed each as a short prompt and pull last hidden
        outs = []
        for t in texts:
            ids = self.processor.tokenizer(t, return_tensors="pt").to(self.device)
            h = self.model.language_model(**ids, output_hidden_states=True).hidden_states[-1].mean(dim=1)
            outs.append(h)
        return torch.cat(outs, dim=0)

# TODO: the text branch by-mean-of-hidden is not how LLaVA was trained.
# results below are *indicative only*.
