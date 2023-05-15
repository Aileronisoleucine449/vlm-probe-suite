"""Base interface for model adapters.

Each adapter wraps a frozen VLM and exposes a small surface:

  - encode_image(images: list[PIL.Image]) -> Tensor [N, D]
  - encode_text(texts: list[str]) -> Tensor [N, D]
  - score(image, text) -> float           (optional; defaults to cosine)

We deliberately do not standardize device handling here -- adapters know best.
"""
from __future__ import annotations
from typing import Iterable
import torch


class ModelAdapter:
    name: str = "unnamed"

    def __init__(self, device: str = "cuda"):
        self.device = device

    def encode_image(self, images: Iterable):  # pragma: no cover
        raise NotImplementedError

    def encode_text(self, texts: Iterable[str]):  # pragma: no cover
        raise NotImplementedError

    def score(self, image, text) -> float:
        img = self.encode_image([image])
        txt = self.encode_text([text])
        img = img / (img.norm(dim=-1, keepdim=True) + 1e-9)
        txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-9)
        return float((img @ txt.T).squeeze().item())
