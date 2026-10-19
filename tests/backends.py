"""Which backends the current environment can run (TF container: tf [+ sigpy]; sigpy container: sigpy)."""

import importlib.util


def available_backends():
    """Backends importable here (``tf`` needs tensorflow-mri, ``sigpy`` needs sigpy)."""
    return [name for name, mod in (("tf", "tensorflow_mri"), ("sigpy", "sigpy"))
            if importlib.util.find_spec(mod) is not None]
