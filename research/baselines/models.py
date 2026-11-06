"""Single-layer LongConv residual trading policy.

The convolution, skip connection, kernel squash, geometric initialization,
hidden size and dropout follow Epstein et al. (2025), Appendix A / Table 4.
The scalar input projection, GELU and scalar readout are explicit reconstruction
choices: the paper does not publish its surrounding network implementation.
Inputs are daily residual returns, not cumulative sums. Only the last causal
convolution output is required to select the next day's holdings.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LongConvPolicy(nn.Module):
    """Map ``[..., lookback]`` lagged residual returns to ``[...]`` scores.

    All assets share the same network. Kernel index zero acts on the most
    recent observation. This module receives history only; the calling
    backtest must pass returns up to t-1 to form the position for return t.
    The returned scores are residual-portfolio allocations. Mapping to stock
    weights and gross-exposure normalization are performed by the backtest.
    """

    def __init__(
        self,
        lookback: int = 30,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        squash: float = 0.001,
    ) -> None:
        super().__init__()
        if lookback < 2 or hidden_dim < 1:
            raise ValueError("lookback must be >= 2 and hidden_dim must be >= 1")
        if squash < 0 or not 0 <= dropout < 1:
            raise ValueError("squash must be nonnegative and dropout in [0, 1)")
        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.squash = squash
        self.input_projection = nn.Linear(1, hidden_dim)
        self.kernel = nn.Parameter(torch.empty(hidden_dim, lookback))
        self.skip = nn.Parameter(torch.ones(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_dim, 1)
        self.reset_kernel_parameters()

    def reset_kernel_parameters(self) -> None:
        """Appendix A geometric initialization (paper's one-based indices)."""
        with torch.no_grad():
            t = torch.arange(1, self.lookback + 1, dtype=self.kernel.dtype,
                             device=self.kernel.device) / self.lookback
            h = torch.arange(1, self.hidden_dim + 1, dtype=self.kernel.dtype,
                             device=self.kernel.device) / self.hidden_dim
            rate = torch.exp(math.log(self.hidden_dim / 2.0) * h)
            self.kernel.copy_(torch.randn_like(self.kernel) *
                              torch.exp(-rate[:, None] * t[None, :]))

    def squashed_kernel(self) -> torch.Tensor:
        """Elementwise soft thresholding, evaluated on every forward pass."""
        return self.kernel.sign() * (self.kernel.abs() - self.squash).clamp_min(0)

    def last_hidden(self, history: torch.Tensor) -> torch.Tensor:
        """Compute the last causal convolution output without large broadcasts.

        Linearity permits folding the scalar embedding into the convolution.
        This is mathematically identical to embedding all L dates and computing
        the last output of a zero-padded FFT convolution, while allocating only
        an [..., hidden_dim] intermediate instead of [..., hidden_dim, L].
        """
        if history.shape[-1] != self.lookback:
            raise ValueError(f"Expected history length {self.lookback}, got {history.shape[-1]}")
        kernel = self.squashed_kernel()
        input_scale = self.input_projection.weight[:, 0]
        weights = kernel.flip(-1) * input_scale[:, None]
        # A new tensor avoids in-place modification of a value used by autograd.
        skip_kernel = F.pad((self.skip * input_scale)[:, None], (self.lookback - 1, 0))
        bias = self.input_projection.bias * (kernel.sum(-1) + self.skip)
        return F.linear(history, weights + skip_kernel, bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.last_hidden(history))
        return self.readout(self.dropout(hidden)).squeeze(-1)
