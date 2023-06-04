"""OpenAI CLIP family via open_clip_torch."""
from __future__ import annotations
import torch
from PIL import Image

try:
    import open_clip
except ImportError:  # pragma: no cover
    open_clip = None

from .base import ModelAdapter


class CLIPAdapter(ModelAdapter):
    def __init__(self, variant: str = "ViT-B-32", pretrained: str = "openai", device: str = "cuda"):
        super().__init__(device=device)
        if open_clip is None:
            raise RuntimeError("install open_clip_torch to use CLIPAdapter")
        self.name = f"clip-{variant}-{pretrained}"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            variant, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(variant)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, images):
        tensors = torch.stack([self.preprocess(img.convert("RGB")) for img in images]).to(self.device)
        return self.model.encode_image(tensors)

    @torch.no_grad()
    def encode_text(self, texts):
        toks = self.tokenizer(list(texts)).to(self.device)
        return self.model.encode_text(toks)
