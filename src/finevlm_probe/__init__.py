"""FineVLM-Probe: a lightweight suite for fine-grained VLM alignment probes."""
from .models import get_model
from .probes import get_probe

__all__ = ["get_model", "get_probe"]
__version__ = "0.6.1"
