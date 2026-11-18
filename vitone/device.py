"""Compute device selection shared by all model-based stages."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def resolve_device(preference: str | None = None) -> torch.device:
    """Return `preference` (e.g. "cpu", "cuda:1") if given, else CUDA, then Apple MPS, then CPU."""
    import torch

    if preference:
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
