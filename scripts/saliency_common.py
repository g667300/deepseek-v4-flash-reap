#!/usr/bin/env python3
"""Small utilities shared by any scoring driver, independent of checkpoint format."""

from __future__ import annotations

from pathlib import Path

import torch


class SaliencyTracker:
    """Accumulates ``sum(g_j * ||f_j||_2)`` and the routed-token count per expert.

    Mirrors ``llmcompressor...reap.utils.REAPSaliencyTracker``: float64
    accumulators, and the reported score is the mean over routed tokens with the
    count clamped to >= 1 so never-routed experts score 0 rather than NaN.
    """

    def __init__(self, num_experts: int, device: torch.device):
        self.num_experts = num_experts
        self.sum = torch.zeros(num_experts, dtype=torch.float64, device=device)
        self.count = torch.zeros(num_experts, dtype=torch.float64, device=device)

    @torch.no_grad()
    def add(self, expert: int, gates: torch.Tensor, norms: torch.Tensor) -> None:
        self.sum[expert] += (gates.to(torch.float64) * norms.to(torch.float64)).sum()
        self.count[expert] += gates.numel()

    @property
    def mean(self) -> torch.Tensor:
        return (self.sum / self.count.clamp(min=1.0)).cpu()

    @property
    def total(self) -> float:
        return float(self.count.sum().item())

    def state(self) -> dict:
        return {"sum": self.sum.cpu(), "count": self.count.cpu()}

    def load(self, state: dict) -> None:
        self.sum = state["sum"].to(self.sum.device)
        self.count = state["count"].to(self.count.device)


def alloc(shape: tuple[int, ...], dtype: torch.dtype, spill: Path | None, name: str) -> torch.Tensor:
    """Allocate a big CPU cache, optionally backed by a file on disk."""
    if spill is None:
        return torch.empty(shape, dtype=dtype)
    spill.mkdir(parents=True, exist_ok=True)
    path = spill / f"{name}.bin"
    numel = 1
    for d in shape:
        numel *= d
    t = torch.from_file(str(path), shared=True, size=numel, dtype=dtype)
    return t.view(*shape)


def human(nbytes: float) -> str:
    return f"{nbytes / 1e9:.1f} GB"


def _rss_gb() -> float:
    """Resident set size, so the reported host use is measured not estimated."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float("nan")
