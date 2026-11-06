"""Independent OU, stock-weight mapping and gradient checks."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch

# run.py is also the directly executable CLI and uses sibling-module imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import numpy_weights, ou_scores, torch_weights
from pipeline import account
from models import LongConvPolicy


def independent_ou(history, valid, entry=1.25, r2_min=0.25):
    """OLS from a design matrix, independent of vectorized moment formulas."""
    answer = []
    for increments, allowed in zip(history, valid):
        if not allowed:
            answer.append(0.0)
            continue
        path = np.cumsum(increments)
        design = np.column_stack([np.ones(len(path)-1), path[:-1]])
        intercept, beta = np.linalg.lstsq(design, path[1:], rcond=None)[0]
        error = path[1:] - design @ [intercept, beta]
        variance = np.mean(error**2)
        tss = np.sum((path[1:]-path[1:].mean())**2)
        r2 = 1-np.sum(error**2)/tss if tss>1e-20 else 0.0
        if not 0<beta<1 or variance<=1e-18 or r2<=r2_min:
            answer.append(0.0)
            continue
        s = (path[-1]-intercept/(1-beta))/np.sqrt(variance/(1-beta**2))
        answer.append(-1.0 if s>entry else (1.0 if s < -entry else 0.0))
    return np.array(answer)


class OUTests(unittest.TestCase):
    @staticmethod
    def ar_path(beta=.75, seed=0, shock=1.0):
        rng = np.random.default_rng(seed)
        path = np.zeros(30)
        for i in range(1, 30):
            path[i] = beta*path[i-1]+rng.normal(scale=.3)
        path[-1] += shock
        return np.diff(np.r_[0.0, path])

    def test_expected_signs_and_invalid_history(self):
        positive = self.ar_path()
        history = np.array([positive, -positive, positive, np.zeros(30)])
        valid = np.array([True, True, False, True])
        result = ou_scores(history, valid)
        np.testing.assert_array_equal(result, [-1, 1, 0, 0])

    def test_nonstationary_negative_beta_and_poor_fit_rejected(self):
        history = np.array([self.ar_path(beta=-.7), self.ar_path(beta=1.15)])
        np.testing.assert_array_equal(ou_scores(history, np.ones(2, bool)), [0, 0])
        # Valid mean reversion with a known tail signal is still rejected by R².
        positive = self.ar_path()
        np.testing.assert_array_equal(ou_scores(positive[None], np.ones(1, bool),
                                                r2_min=.999999), [0])

    def test_vectorized_rule_matches_independent_least_squares(self):
        history = np.array([self.ar_path(beta=b, seed=seed, shock=shock)
                            for b in [.3, .75, 1.1, -.5]
                            for seed in range(6) for shock in [-1., 0., 1.]])
        valid = np.ones(len(history), bool)
        valid[::7] = False
        for entry in [1.0, 1.25, 1.5]:
            for r2_min in [.25, .5, .75]:
                expected = independent_ou(history, valid, entry, r2_min)
                np.testing.assert_array_equal(ou_scores(history, valid, entry, r2_min), expected)


class GivenScores(torch.nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.scores = scores

    def forward(self, unused):
        return self.scores


class StockMappingTests(unittest.TestCase):
    def fixture(self, dtype=np.float64):
        rng = np.random.default_rng(734)
        b, n, k = 3, 5, 2
        basis = np.zeros((b, n, k), dtype=dtype)
        basis[:, :4] = np.stack([np.linalg.qr(rng.normal(size=(4, k)))[0] for _ in range(b)])
        ids = np.array([[4, 0, 2, 5, -1], [2, 4, 5, 0, -1], [0, 4, 2, 5, -1]])
        scales = rng.uniform(.4, 3, size=(b, n)).astype(dtype)
        q = rng.normal(size=(b, n)).astype(dtype)
        q[:, -1] = 0
        valid = ids>=0
        data = {'basis': basis, 'scales': scales, 'ids': ids, 'permnos': np.arange(6)}
        batch = {'x': torch.zeros((b, n, 30), dtype=torch.float64),
                 'valid': torch.tensor(valid), 'ids': torch.tensor(ids),
                 'v': torch.tensor(basis), 's': torch.tensor(scales)}
        return q, data, batch, k

    def test_numpy_and_torch_match_explicit_transposed_projection(self):
        q, data, batch, k = self.fixture()
        expected = np.zeros((len(q), len(data['permnos'])))
        for t in range(len(q)):
            v, s = data['basis'][t], data['scales'][t]
            p = np.eye(len(s))-np.diag(s) @ v @ v.T @ np.diag(1/s)
            stock = p.T @ q[t]
            stock /= np.abs(stock).sum()
            active = data['ids'][t]>=0
            expected[t, data['ids'][t, active]] = stock[active]
        actual_numpy = numpy_weights(q, np.arange(len(q)), data, k)
        actual_torch = torch_weights(GivenScores(torch.tensor(q)), batch, 6).detach().numpy()
        np.testing.assert_allclose(actual_numpy, expected, atol=1e-7, rtol=1e-6)
        np.testing.assert_allclose(actual_torch, expected, atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(np.abs(actual_torch).sum(-1), 1)
        np.testing.assert_array_equal(actual_torch[:, [1, 3]], 0)

    def test_weight_mapping_gradients_match_finite_differences(self):
        q, data, batch, k = self.fixture()
        variable = torch.tensor(q, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda values: torch_weights(GivenScores(values), batch, 6),
            (variable,), eps=1e-6, atol=2e-5, rtol=1e-3))

    def test_training_objective_costs_match_backtest_accounting(self):
        q, data, batch, k = self.fixture()
        rng = np.random.default_rng(44)
        returns = rng.normal(scale=.02, size=(3, 6))
        weights = torch_weights(GivenScores(torch.tensor(q)), batch, 6)
        gross = (weights*torch.tensor(returns)).sum(-1)
        previous = torch.cat([torch.zeros_like(weights[:1]), weights[:-1]], dim=0)
        cost = .0005*(weights-previous).abs().sum(-1)+.0001*(-weights).clamp_min(0).sum(-1)
        accounting_data = {
            'returns': returns, 'returns_missing': np.zeros_like(returns, dtype=bool),
            'rf': np.zeros(3), 'market_proxy': np.zeros(3),
            'calendar': pd.bdate_range('2000-12-29', periods=3)}
        frame = account(weights.detach().numpy(), accounting_data, np.arange(3),
                        {'transaction_cost': .0005, 'short_cost': .0001})
        np.testing.assert_allclose((gross-cost).detach().numpy(), frame.net_excess, atol=1e-14)

    def test_longconv_gets_finite_nonzero_gradients_through_stock_mapping(self):
        torch.manual_seed(12)
        q, data, batch, k = self.fixture()
        batch['x'] = torch.randn((3, 5, 30), dtype=torch.float64)*.02
        model = LongConvPolicy(dropout=0).double()
        weights = torch_weights(model, batch, 6)
        returns = torch.randn((3, 6), dtype=torch.float64)*.02
        previous = torch.cat([torch.zeros_like(weights[:1]), weights[:-1]], dim=0)
        net = (weights*returns).sum(-1)-.0005*(weights-previous).abs().sum(-1)
        net = net-.0001*(-weights).clamp_min(0).sum(-1)
        loss = -net.mean()/net.std(unbiased=False)
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.kernel.grad.abs().sum().item(), 0)
        self.assertGreater(model.input_projection.weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
