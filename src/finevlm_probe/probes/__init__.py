"""Probe registry."""
from .attr_swap import AttrSwapProbe
from .count_coarse import CountCoarseProbe

_PROBES = {
    AttrSwapProbe.name: AttrSwapProbe,
    CountCoarseProbe.name: CountCoarseProbe,
}


def get_probe(name: str):
    if name not in _PROBES:
        raise KeyError(f"unknown probe: {name}. known: {sorted(_PROBES)}")
    return _PROBES[name]()
