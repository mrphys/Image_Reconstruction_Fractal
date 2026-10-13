
"""A small framework for generating quaternion fractals and complex/multichannel images."""

from .core import (
    QuaternionJulia,
    JuliaConfig,
    Slice,
    FieldSpec,
    render_fields,
)
from .fields import FIELD_REGISTRY, register_field
from .channels import (
    normalise,
    remap,
    compose_channel,
    compose_complex,
    complex_channels,
    stack_channels,
)
from .warp import domain_warp
from .mandelbrot import mandelbrot_escape, mandelbrot_grid_4d, candidate_c

__all__ = [
    "QuaternionJulia", "JuliaConfig", "Slice", "FieldSpec", "render_fields",
    "FIELD_REGISTRY", "register_field",
    "normalise", "remap", "compose_channel", "compose_complex",
    "complex_channels", "stack_channels", "domain_warp",
    "mandelbrot_escape", "mandelbrot_grid_4d", "candidate_c",
]
