"""Model adapter registry."""
from typing import Dict, Type
from .base import ModelAdapter

_REGISTRY: Dict[str, Type[ModelAdapter]] = {}


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def get_model(name: str, **kwargs) -> ModelAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model: {name}. registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


# Built-ins (lazy: only loaded on first construct, not import)
def _init_builtins():
    try:
        from .clip_adapter import CLIPAdapter
        register("clip_vit_b32")(lambda **kw: CLIPAdapter("ViT-B-32", "openai", **kw))
        register("clip_vit_b16")(lambda **kw: CLIPAdapter("ViT-B-16", "openai", **kw))
        register("clip_vit_l14")(lambda **kw: CLIPAdapter("ViT-L-14", "openai", **kw))
        register("clip_vit_l14_336")(lambda **kw: CLIPAdapter("ViT-L-14-336", "openai", **kw))
    except Exception:
        pass
    try:
        from .siglip_adapter import SigLIPAdapter
        register("siglip_base_224")(lambda **kw: SigLIPAdapter("google/siglip-base-patch16-224", **kw))
        register("siglip_large_384")(lambda **kw: SigLIPAdapter("google/siglip-large-patch16-384", **kw))
        register("siglip_so400m_384")(lambda **kw: SigLIPAdapter("google/siglip-so400m-patch14-384", **kw))
    except Exception:
        pass


_init_builtins()

# blip2 registered manually below since the import is heavy
try:
    from .blip2_adapter import BLIP2Adapter  # noqa
    register('blip2_itm_vit_g')(lambda **kw: BLIP2Adapter(**kw))
except Exception:
    pass
