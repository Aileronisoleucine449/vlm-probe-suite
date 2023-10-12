import pytest
from finevlm_probe.probes import get_probe
from finevlm_probe.probes.attr_swap import AttrSwapProbe
from finevlm_probe.probes.count_coarse import CountCoarseProbe


def test_get_probe_known():
    assert isinstance(get_probe("attr-swap"), AttrSwapProbe)
    assert isinstance(get_probe("count-coarse"), CountCoarseProbe)


def test_get_probe_unknown():
    with pytest.raises(KeyError):
        get_probe("not-a-thing")


def test_attr_swap_summarize_empty():
    p = AttrSwapProbe()
    s = p.summarize([])
    assert s["n"] == 0


def test_attr_swap_summarize():
    p = AttrSwapProbe()
    recs = [
        {"sample_id": "a", "correct": 1, "margin": 0.1},
        {"sample_id": "b", "correct": 0, "margin": -0.2},
        {"sample_id": "c", "correct": 1, "margin": 0.05},
    ]
    s = p.summarize(recs)
    assert s["n"] == 3
    assert abs(s["accuracy"] - 2/3) < 1e-6


def test_count_coarse_bucket_summarize():
    p = CountCoarseProbe()
    recs = [
        {"gold": "1", "pred": "1", "correct": 1},
        {"gold": "3", "pred": "1", "correct": 0},
        {"gold": "5+", "pred": "5+", "correct": 1},
    ]
    s = p.summarize(recs)
    assert 0.0 <= s["macro_f1"] <= 1.0
    assert s["accuracy"] == pytest.approx(2/3)
