"""Independent numerical/causality/accounting tests using tiny synthetic data.

Run: python3 -m unittest discover -s tests -v
No supplied parquet file is read by these tests.
"""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline


class PCAMathematicsTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(71)
        self.history = (self.rng.normal(size=(15, 8)) * np.arange(1, 9)
                        + np.linspace(-3, 4, 8))

    def test_factor_span_matches_independent_full_svd(self):
        for mode in ['covariance', 'correlation']:
            with self.subTest(mode=mode):
                basis, scales = pipeline.pca_basis(self.history, 3, mode)
                centered = self.history-self.history.mean(axis=0)
                if mode == 'correlation':
                    centered = centered/centered.std(axis=0, ddof=1)
                _, _, right = np.linalg.svd(centered, full_matrices=False)
                reference = right[:3].T
                # Eigenvector signs are arbitrary; compare the subspaces instead.
                np.testing.assert_allclose(basis @ basis.T, reference @ reference.T,
                                           atol=2e-7, rtol=2e-6)
                np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=2e-7)

    def test_score_mapping_preserves_portfolio_return(self):
        returns = self.rng.normal(size=8)
        scores = self.rng.normal(size=8)
        for mode in ['covariance', 'correlation']:
            with self.subTest(mode=mode):
                basis, scales = pipeline.pca_basis(self.history, 3, mode)
                residuals = pipeline.residualize(returns, basis, scales)
                stock_weights = pipeline.map_scores(scores, basis, scales)
                np.testing.assert_allclose(scores @ residuals,
                                           stock_weights @ returns,
                                           atol=2e-7, rtol=2e-6)
                # Hedged weights have zero exposure to the estimated factor loadings.
                np.testing.assert_allclose(stock_weights @ (scales[:, None]*basis),
                                           np.zeros(3), atol=2e-6)
        basis, scales = pipeline.pca_basis(self.history, 3, 'correlation')
        # Nonuniform volatilities make P asymmetric; P' is essential for weights.
        p = np.eye(8)-scales[:, None]*(basis @ basis.T)/scales[None, :]
        self.assertGreater(np.linalg.norm(p-p.T), .1)

    def test_normalized_weights_and_zero_portfolio(self):
        weights = np.array([[3., -1., 0.], [0., 0., 0.], [-2., -5., 3.]])
        result = pipeline.normalize_weights(weights)
        np.testing.assert_allclose(np.abs(result).sum(axis=1), [1, 0, 1])
        np.testing.assert_array_equal(result[1], np.zeros(3))
        np.testing.assert_allclose(result[0], [.75, -.25, 0])
        np.testing.assert_allclose(result, pipeline.normalize_weights(10*weights))


class InputCausalityTests(unittest.TestCase):
    def test_signal_history_uses_lagged_permanent_identifiers(self):
        ids = np.array([[0, 1, -1], [1, 0, -1], [2, 1, -1],
                        [1, 2, -1], [2, 0, -1], [0, 1, -1]], dtype=np.int32)
        residuals = np.arange(18, dtype=np.float32).reshape(6, 3)
        residuals[1, 2] = np.nan
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path/'complete.json').write_text(json.dumps({'fingerprint': 'synthetic-parent'}))
            np.save(path/'residuals_k1.npy', residuals)
            x, valid = pipeline.make_inputs({'path': path, 'ids': ids}, 1, 2)
            np.testing.assert_array_equal(x[4, 0], residuals[2:4, 2])
            np.testing.assert_array_equal(x[4, 1], residuals[2:4, 0])
            self.assertFalse(valid[2, 0])  # ID 2 has a missing historical observation.
            self.assertTrue(valid[4, 0])
            self.assertTrue(valid[4, 1])
            self.assertFalse(valid[:, 2].any())  # Padding never becomes tradable.
            self.assertFalse(valid[:2].any())  # Initial incomplete histories stay invalid.
            before = np.asarray(x[:5]).copy()
        # Changing date t and later returns must leave t's signal unchanged.
        altered = residuals.copy()
        altered[4:] = -999
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path/'complete.json').write_text(json.dumps({'fingerprint': 'synthetic-parent'}))
            np.save(path/'residuals_k1.npy', altered)
            x, valid = pipeline.make_inputs({'path': path, 'ids': ids}, 1, 2)
            np.testing.assert_array_equal(before, x[:5])

    def test_interrupted_input_cache_without_marker_is_rebuilt(self):
        ids = np.tile(np.arange(2), (5, 1)).astype(np.int32)
        residuals = np.arange(10, dtype=np.float32).reshape(5, 2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path/'complete.json').write_text(json.dumps({'fingerprint': 'parent-A'}))
            np.save(path/'residuals_k1.npy', residuals)
            # Simulate files that exist after a failed generation, with invalid contents.
            np.save(path/'inputs_k1_l2.npy', np.full((5, 2, 2), 999, dtype=np.float32))
            np.save(path/'input_valid_k1_l2.npy', np.ones((5, 2), dtype=bool))
            x, valid = pipeline.make_inputs({'path': path, 'ids': ids}, 1, 2)
            np.testing.assert_array_equal(x[4, 0], residuals[2:4, 0])
            self.assertFalse(valid[:2].any())
            self.assertTrue(valid[2:].all())
            marker = json.loads((path/'input_complete_k1_l2.json').read_text())
            self.assertEqual(marker['fingerprint'], 'parent-A')

    def test_input_cache_tracks_parent_fingerprint(self):
        ids = np.tile(np.arange(2), (5, 1)).astype(np.int32)
        residuals = np.arange(10, dtype=np.float32).reshape(5, 2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path/'complete.json').write_text(json.dumps({'fingerprint': 'parent-A'}))
            np.save(path/'residuals_k1.npy', residuals)
            x, valid = pipeline.make_inputs({'path': path, 'ids': ids}, 1, 2)
            del x, valid
            # A completed but stale input cache must not survive a new PCA parent.
            (path/'complete.json').write_text(json.dumps({'fingerprint': 'parent-B'}))
            np.save(path/'residuals_k1.npy', residuals+123)
            x, valid = pipeline.make_inputs({'path': path, 'ids': ids}, 1, 2)
            np.testing.assert_array_equal(x[4, 0], residuals[2:4, 0]+123)
            marker = json.loads((path/'input_complete_k1_l2.json').read_text())
            self.assertEqual(marker['fingerprint'], 'parent-B')

    def test_prepare_pca_excludes_current_and_future_returns(self):
        rng = np.random.default_rng(19)
        dates = pd.bdate_range('2001-01-01', periods=12)
        permnos = np.array([11, 22, 33, 44, 55, 66])
        returns = rng.normal(scale=.01, size=(len(dates), len(permnos)))
        features = pd.DataFrame({'DATE': dates, 'rf': .0001})
        universe = pd.DataFrame({'PERMNO': permnos,
                                 'eval_month': pd.Timestamp('2001-01-31')})
        target = 7

        def run(values, folder):
            panel = pd.DataFrame({'DATE': np.repeat(dates, len(permnos)),
                                  'PERMNO': np.tile(permnos, len(dates)),
                                  'RET_ADJ': values.ravel(),
                                  'DLRET': np.full(values.size, np.nan),
                                  'me': np.tile(np.arange(1., 7.), len(dates))})
            cfg = {'daily_panel': 'daily', 'features': 'features', 'universe_monthly': 'universe',
                   'cache_start': str(dates[4].date()), 'pca_window': 4,
                   'factor_counts': [1], 'lookback': 2, 'pca_mode': 'covariance', 'threads': 1}
            frames = {'daily': panel, 'features': features, 'universe': universe}

            def fake_read(path, columns=None):
                result = frames[path].copy()
                return result if columns is None else result[columns]

            with patch.object(pipeline, 'fingerprint', return_value='synthetic'), \
                    patch.object(pipeline.pd, 'read_parquet', side_effect=fake_read):
                data = pipeline.prepare(cfg, cache=folder)
            i = target-4
            v = np.array(data['basis'][i])
            r = np.load(Path(folder)/'residuals_k1.npy')[i].copy()
            return v @ v.T, r

        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            projection_before, residual_before = run(returns, one)
            altered = returns.copy()
            altered[target:] += rng.normal(size=altered[target:].shape)
            projection_after, residual_after = run(altered, two)
            np.testing.assert_array_equal(projection_before, projection_after)
            self.assertGreater(np.linalg.norm(residual_before-residual_after), .01)

    def test_delisting_known_yesterday_removes_stock_today(self):
        dates = pd.bdate_range('2001-01-01', periods=9)
        permnos = np.array([11, 22, 33, 44])
        rng = np.random.default_rng(919)
        panel = pd.DataFrame({'DATE': np.repeat(dates, len(permnos)),
                              'PERMNO': np.tile(permnos, len(dates)),
                              'RET_ADJ': rng.normal(scale=.02, size=len(dates)*len(permnos)),
                              'DLRET': np.full(len(dates)*len(permnos), np.nan),
                              'me': 1.0})
        panel.loc[(panel.DATE==dates[4]) & (panel.PERMNO==11), 'DLRET'] = -.3
        features = pd.DataFrame({'DATE': dates, 'rf': .0001})
        universe = pd.DataFrame({'PERMNO': permnos, 'eval_month': pd.Timestamp('2001-01-31')})
        cfg = {'daily_panel':'daily', 'features':'features', 'universe_monthly':'universe',
               'cache_start':str(dates[3].date()), 'pca_window':3, 'factor_counts':[1],
               'lookback':2, 'pca_mode':'covariance', 'threads':1}
        frames = {'daily':panel, 'features':features, 'universe':universe}
        def fake_read(path, columns=None):
            return frames[path].copy() if columns is None else frames[path][columns].copy()
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(pipeline, 'fingerprint', return_value='delisting-fixture'), \
                patch.object(pipeline.pd, 'read_parquet', side_effect=fake_read):
            data=pipeline.prepare(cfg,cache=folder)
            self.assertIn(0, data['ids'][1])  # Event-day return is still earned.
            self.assertNotIn(0, data['ids'][2])  # Public delisting excludes next day.

    def test_offline_cache_rejects_changed_configuration_and_files(self):
        import hashlib
        cfg={'cache_start':'1989-01-01','pca_window':252,'lookback':30,
             'factor_counts':[1], 'pca_mode':'covariance',
             'daily_panel':'a','features':'b','universe_monthly':'c'}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            (path/'complete.json').write_text(json.dumps({'fingerprint':'abc'}))
            (path/'dummy.npy').write_bytes(b'original')
            seal={'fingerprint':'abc','definition':pipeline.cache_definition(cfg),
                  'sha256':{'dummy.npy':hashlib.sha256(b'original').hexdigest()}}
            (path/'seal.json').write_text(json.dumps(seal))
            altered=dict(cfg,pca_window=120)
            with self.assertRaisesRegex(RuntimeError,'configuration'):
                pipeline.load_sealed_cache(altered,path)
            (path/'dummy.npy').write_bytes(b'damaged')
            with self.assertRaisesRegex(RuntimeError,'changed'):
                pipeline.load_sealed_cache(cfg,path)


class PortfolioAccountingTests(unittest.TestCase):
    def setUp(self):
        self.weights = np.array([[.6, -.4, 0.], [0., .5, -.5],
                                 [.2, .3, -.5], [0., 0., 0.]], dtype=np.float64)
        self.returns = np.array([[.01, -.02, .3], [-.01, .03, .04],
                                 [.02, .01, -.04], [.1, -.1, .1]])
        self.data = {'returns': self.returns,
                     'returns_missing': np.zeros((4, 3), dtype=bool),
                     'calendar': pd.DatetimeIndex(['2000-12-28', '2000-12-29', '2001-01-02', '2001-01-03']),
                     'rf': np.repeat(.0001, 4), 'market_proxy': np.zeros(4)}
        self.cfg = {'transaction_cost': .0005, 'short_cost': .0001}

    def test_entries_exits_and_cost_components(self):
        f = pipeline.account(self.weights, self.data, np.arange(4), self.cfg)
        expected_turnover = np.array([1., 2., .4, 1.])
        expected_short = np.array([.4, .5, .5, 0.])
        expected_gross = np.array([.014, -.005, .027, 0.])
        np.testing.assert_allclose(f.turnover_l1, expected_turnover)
        np.testing.assert_allclose(f.short_exposure, expected_short)
        np.testing.assert_allclose(f.gross_excess, expected_gross)
        np.testing.assert_allclose(f.net_excess,
                                   expected_gross-.0005*expected_turnover-.0001*expected_short)
        self.assertAlmostEqual(f.gross_exposure.iloc[3], 0)
        self.assertAlmostEqual(f.transaction_cost.iloc[3], .0005)  # Closing costs money.

    def test_year_splits_equal_continuous_accounting_with_previous_weights(self):
        whole = pipeline.account(self.weights, self.data, np.arange(4), self.cfg)
        year_one = pipeline.account(self.weights[:2], self.data, np.array([0, 1]), self.cfg)
        year_two = pipeline.account(self.weights[2:], self.data, np.array([2, 3]), self.cfg,
                                    previous=self.weights[1])
        split = pd.concat([year_one, year_two], ignore_index=True)
        pd.testing.assert_frame_equal(whole, split)
        incorrectly_reset = pipeline.account(self.weights[2:], self.data, np.array([2, 3]), self.cfg)
        self.assertAlmostEqual(year_two.turnover_l1.iloc[0], .4)
        self.assertAlmostEqual(incorrectly_reset.turnover_l1.iloc[0], 1.)

    def test_missing_return_exposure_is_explicit(self):
        self.data['returns_missing'][2, 2] = True
        self.data['returns'][2, 2] = 0  # The pipeline's disclosed zero-excess valuation convention.
        f = pipeline.account(self.weights, self.data, np.arange(4), self.cfg)
        np.testing.assert_allclose(f.missing_return_exposure, [0, 0, .5, 0])
        self.assertAlmostEqual(f.gross_excess.iloc[2], .007)


if __name__ == '__main__':
    unittest.main()
