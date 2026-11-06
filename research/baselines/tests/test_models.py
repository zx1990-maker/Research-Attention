"""Numerical and causality checks for the reconstructed LongConv layer."""

import unittest
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import LongConvPolicy


def fft_reference_hidden(model, history):
    """Independent full sequence, zero-padded causal convolution reference."""
    embedded = model.input_projection(history.unsqueeze(-1)).transpose(-1, -2)
    length = history.shape[-1]
    kernel = model.squashed_kernel()
    fft_length = 2 * length
    convolution = torch.fft.irfft(
        torch.fft.rfft(embedded, n=fft_length, dim=-1)
        * torch.fft.rfft(kernel, n=fft_length, dim=-1),
        n=fft_length, dim=-1,
    )[..., :length]
    return convolution + model.skip[:, None] * embedded


class LongConvTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = LongConvPolicy(lookback=30, hidden_dim=32, dropout=0.1).double().eval()
        self.history = torch.randn(3, 7, 30, dtype=torch.float64) * 0.02

    def test_final_output_matches_zero_padded_fft(self):
        reference = fft_reference_hidden(self.model, self.history)[..., -1]
        torch.testing.assert_close(self.model.last_hidden(self.history), reference,
                                   atol=1e-12, rtol=1e-10)
        scores = self.model.readout(F.gelu(reference)).squeeze(-1)
        torch.testing.assert_close(self.model(self.history), scores,
                                   atol=1e-12, rtol=1e-10)

    def test_gradients_match_fft_reference(self):
        parameters = tuple(self.model.parameters())
        self.model(self.history).square().sum().backward()
        direct_grads = [p.grad.detach().clone() for p in parameters]
        self.model.zero_grad(set_to_none=True)
        reference = fft_reference_hidden(self.model, self.history)[..., -1]
        scores = self.model.readout(F.gelu(reference)).squeeze(-1)
        scores.square().sum().backward()
        for actual, p in zip(direct_grads, parameters):
            torch.testing.assert_close(actual, p.grad, atol=1e-10, rtol=1e-8)

    def test_convolution_does_not_read_future_observations(self):
        altered = self.history.clone()
        altered[..., 15:] += 100.0
        original_y = fft_reference_hidden(self.model, self.history)[..., :15]
        altered_y = fft_reference_hidden(self.model, altered)[..., :15]
        torch.testing.assert_close(original_y, altered_y, atol=1e-11, rtol=1e-10)

    def test_asset_permutation_equivariance(self):
        order = torch.tensor([5, 2, 6, 1, 0, 4, 3])
        torch.testing.assert_close(self.model(self.history[:, order]),
                                   self.model(self.history)[:, order])

    def test_squash_and_input_length(self):
        with torch.no_grad():
            self.model.kernel.fill_(0.0005)
        self.assertEqual(torch.count_nonzero(self.model.squashed_kernel()).item(), 0)
        with self.assertRaises(ValueError):
            self.model(self.history[..., :-1])


if __name__ == "__main__":
    unittest.main()
