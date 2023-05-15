"""Probe protocol."""
from __future__ import annotations
from typing import Iterable, Any


class Probe:
    name: str = "unnamed"

    def prepare(self, dataset):
        raise NotImplementedError

    def step(self, sample, model) -> dict:
        raise NotImplementedError

    def summarize(self, records: Iterable[dict]) -> dict:
        raise NotImplementedError
