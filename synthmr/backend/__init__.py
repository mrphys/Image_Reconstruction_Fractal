"""Backend selection.

A backend is a module (package) exposing the functions listed in docs/BACKEND.md.
Library code calls ``get_backend()`` instead of importing an array/MRI library.
"""

import importlib

_REGISTRY = {"tf": "synthmr.backend.tf"}
_active = None


def register_backend(name, module_path):
    """Make a backend importable by name, e.g. ``register_backend("torch", "synthmr.backend.torch")``."""
    _REGISTRY[name] = module_path


def set_backend(name):
    """Activate backend ``name`` and return its module."""
    global _active
    if name not in _REGISTRY:
        raise KeyError(f"Unknown backend {name!r}. Available: {sorted(_REGISTRY)}")
    _active = importlib.import_module(_REGISTRY[name])
    return _active


def get_backend():
    """Return the active backend module (TensorFlow by default)."""
    return _active if _active is not None else set_backend("tf")
