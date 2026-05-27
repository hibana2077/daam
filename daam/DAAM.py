"""Backward-compatible imports for old `daam.DAAM` users."""

from .core import DAAMResult, TimmViTDAAM, create_daam, list_timm_vit_models

DAAM = TimmViTDAAM

__all__ = [
    "DAAM",
    "DAAMResult",
    "TimmViTDAAM",
    "create_daam",
    "list_timm_vit_models",
]
