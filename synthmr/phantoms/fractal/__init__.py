"""Quaternion fractal fields and grids (moved from ``app/fractal_framework``)."""

from .channels import (
    balance,
    complex_channels,
    compose_channel,
    compose_complex,
    normalise,
    remap,
    stack_channels,
)
from .core import FieldSpec, JuliaConfig, QuaternionJulia, Slice, render_fields
from .fields import FIELD_REGISTRY, evaluate_field, register_field
from .grid import Axis, Grid
from .mandelbrot import candidate_c, mandelbrot_escape, mandelbrot_grid_4d
from .warp import domain_warp

__all__ = [
    "QuaternionJulia", "JuliaConfig", "Slice", "FieldSpec", "render_fields",
    "FIELD_REGISTRY", "register_field", "evaluate_field", "Axis", "Grid",
    "normalise", "balance", "remap", "compose_channel", "compose_complex",
    "complex_channels", "stack_channels", "domain_warp",
    "mandelbrot_escape", "mandelbrot_grid_4d", "candidate_c",
]
