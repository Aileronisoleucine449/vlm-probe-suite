"""SigLIP via HF transformers."""
from __future__ import annotations
import torch
from .base import ModelAdapter

try:
    from transformers import AutoModel, AutoProcessor
except ImportError:  # pragma: no cover
    AutoModel = None
    AutoProcessor = None


class SigLIPAdapter(ModelAdapter):
    def __init__(self, hf_id: str = "google/siglip-base-patch16-224", device: str = "cuda"):
        super().__init__(device=device)
        if AutoModel is None:
            raise RuntimeError("install transformers to use SigLIPAdapter")
        self.name = f"siglip-{hf_id.split('/')[-1]}"
        self.model = AutoModel.from_pretrained(hf_id).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(hf_id)

    @torch.no_grad()
    def encode_image(self, images):
        inp = self.processor(images=list(images), return_tensors="pt").to(self.device)
        return self.model.get_image_features(**inp)

    @torch.no_grad()
    def encode_text(self, texts):
        inp = self.processor(text=list(texts), padding="max_length",
                             return_tensors="pt").to(self.device)
        return self.model.get_text_features(**inp)
