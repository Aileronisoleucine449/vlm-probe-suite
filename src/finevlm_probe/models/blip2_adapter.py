"""BLIP-2 adapter, ITM-head mode.

We use the image-text-matching head logits as the score, not generation
likelihood. See report Appendix C for the why.
"""
from __future__ import annotations
import torch
from .base import ModelAdapter

try:
    from transformers import Blip2ForImageTextRetrieval, Blip2Processor
except ImportError:  # pragma: no cover
    Blip2ForImageTextRetrieval = None
    Blip2Processor = None


class BLIP2Adapter(ModelAdapter):
    def __init__(self, hf_id: str = "Salesforce/blip2-itm-vit-g", device: str = "cuda"):
        super().__init__(device=device)
        if Blip2ForImageTextRetrieval is None:
            raise RuntimeError("transformers >=4.40 needed for Blip2ForImageTextRetrieval")
        self.name = f"blip2-{hf_id.split('/')[-1]}"
        self.processor = Blip2Processor.from_pretrained(hf_id)
        self.model = Blip2ForImageTextRetrieval.from_pretrained(hf_id).to(device).eval()

    @torch.no_grad()
    def score(self, image, text) -> float:
        inp = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        out = self.model(**inp, use_image_text_matching_head=True)
        # ITM logits: [batch, 2] -- take softmax'd "match" prob
        prob = out.logits_per_image.softmax(dim=-1)[..., 1].item()
        return prob

    def encode_image(self, images):
        raise NotImplementedError("BLIP2Adapter uses score() directly")

    def encode_text(self, texts):
        raise NotImplementedError("BLIP2Adapter uses score() directly")
