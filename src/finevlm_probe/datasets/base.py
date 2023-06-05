"""Dataset loader interfaces."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator
from PIL import Image


@dataclass
class PairSample:
    id: str
    image: Image.Image
    positive_caption: str
    negative_caption: str


@dataclass
class CountSample:
    id: str
    image: Image.Image
    object_class: str
    count: int


class Dataset:
    name: str = "unnamed"

    def iter_pairs(self, probe_name: str) -> Iterator[PairSample]:
        raise NotImplementedError

    def iter_count_samples(self) -> Iterator[CountSample]:
        raise NotImplementedError
