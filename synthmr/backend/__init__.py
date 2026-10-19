"""Backend selection.

A backend is a module (package) exposing the functions listed in docs/BACKEND.md.
Library code calls ``get_backend()`` instead of importing an array/MRI library.
The default backend is ``$SYNTHMR_BACKEND`` if set, else ``tf``.
"""

import importlib
import os

_REGISTRY = {"tf": "synthmr.backend.tf", "sigpy": "synthmr.backend.sigpy"}
_active = None


def register_backend(name, module_path):
    """Make a backend importable by name, e.g. ``register_backend("torch", "synthmr.backend.torch")``."""
    _REGISTRY[name] = module_path


def load_backend(name):
    """Return backend module ``name`` without making it the active (default) backend."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown backend {name!r}. Available: {sorted(_REGISTRY)}")
    return importlib.import_module(_REGISTRY[name])


def set_backend(name):
    """Activate backend ``name`` and return its module."""
    global _active
    _active = load_backend(name)
    return _active


def get_backend():
    """Return the active backend module (``$SYNTHMR_BACKEND``, default TensorFlow)."""
    return _active if _active is not None else set_backend(os.environ.get("SYNTHMR_BACKEND", "tf"))
