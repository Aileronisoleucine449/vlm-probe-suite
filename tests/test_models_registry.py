import pytest
from finevlm_probe.models import get_model, _REGISTRY


def test_unknown_model():
    with pytest.raises(KeyError):
        get_model("nope")


def test_some_models_registered():
    # at least the lambdas registered, even if loading fails when called
    assert len(_REGISTRY) >= 1
