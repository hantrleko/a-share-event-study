"""V1 checks: fast placebos, inference extras, isolation, and manifests."""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare_event.audit import audit_grid
from ashare_event.config import load_config, validate_windows
from ashare_event.diagnostics import variance_inflation
from ashare_event.inference import bh_adjust
from ashare_event.methods_text import render_methods
from ashare_event.pipeline import run_study
from ashare_event.placebo import event_cars, run_placebos
from ashare_event.regression import fit_car_regressions
from ashare_event.returns import _returns_and_calendar, add_peer_count, estimate_events, is_limit_hit, limit_band
from test_core import synthetic


def _cfg():
    cfg = load_config(ROOT / "config" / "study.yml")
    cfg.estimation_window = (-80, -16)
    cfg.event_windows = [(0, 0), (-1, 1)]
    cfg.path_window = (-5, 5)
    cfg.regression_window = (0, 0)
    cfg.min_estimation_obs = 40
    cfg.placebo_draws = 0
    cfg.placebo_modes = ["in_time"]
    cfg.placebo_offsets = (-40, -20)
    cfg.two_way_cluster = False
    cfg.wild_draws = 0
    cfg.verbose = False
    cfg.model = "market"
    validate_windows(cfg)
    return cfg


def _estimate(events, prices, market, cfg, **kwargs):
    return estimate_events(
        events,
        prices,
        market,
        estimation_window=cfg.estimation_window,
        event_windows=cfg.event_windows,
        path_window=cfg.path_window,
        regression_window=cfg.regression_window,
        model=kwargs.pop("model", cfg.model),
        min_estimation_obs=cfg.min_estimation_obs,
        **kwargs,
    )


class UpgradeTests(unittest.TestCase):
    def test_fast_car_matches_the_refit(self):
        events, prices, market, _dates = synthetic(
            n_firms=6,
            shocks=[0.02] * 6,
            eps_sd=0.0,
            event_locs=[150 + i for i in range(6)],
        )
        cfg = _cfg()
        slow = _estimate(events, prices, market, cfg)
        fast = event_cars(events, prices, market, cfg)
        slow_car = slow.events.set_index("event_id")["car_p0_p0"]
        np.testing.assert_allclose(fast.reindex(slow_car.index).to_numpy(dtype=float), slow_car.to_numpy(dtype=float), atol=1e-8)
        _by_code, _mkt, calendar = _returns_and_calendar(prices, market)
        shifted = events.copy()
        shifted["event_date"] = [
            calendar[int(calendar.get_loc(pd.Timestamp(day).normalize())) - 25] for day in shifted["event_date"]
        ]
        slow_shift = _estimate(shifted, prices, market, cfg)
        fast_shift = event_cars(shifted, prices, market, cfg)
        slow_values = slow_shift.events.set_index("event_id")["car_p0_p0"]
        np.testing.assert_allclose(
            fast_shift.reindex(slow_values.index).to_numpy(dtype=float),
            slow_values.to_numpy(dtype=float),
            atol=1e-8,
        )

    def test_after_close_skips_the_announcement_day(self):
        events, prices, market, dates = synthetic(event_locs=[180], shocks=[0.04], eps_sd=0.0)
        events["after_close"] = True
        estimated = _estimate(events, prices, market, _cfg())
        self.assertEqual(pd.Timestamp(estimated.events["event_date_aligned"].iloc[0]), dates[181])
        self.assertTrue(bool(estimated.events["rolled_forward"].iloc[0]))
        self.assertAlmostEqual(float(estimated.events["car_p0_p0"].iloc[0]), 0.0, places=6)

    def test_limit_flag_uses_the_raw_return(self):
        self.assertEqual(limit_band("000001", ""), 0.10)
        self.assertEqual(limit_band("300001", ""), 0.20)
        self.assertEqual(limit_band("600000", "ST 示例"), 0.05)
        self.assertTrue(is_limit_hit(0.099, 0.10))
        self.assertFalse(is_limit_hit(0.02, 0.10))
        dates = pd.bdate_range("2020-01-06", periods=140)
        rets = np.zeros(140)
        rets[90] = 0.099
        prices = pd.DataFrame({"code": "000001", "date": dates, "close": 10 * np.cumprod(1.0 + rets)})
        market = pd.DataFrame({"date": dates, "close": np.full(140, 1000.0)})
        events = pd.DataFrame(
            {
                "firm_id": ["F1"],
                "code": ["000001"],
                "event_date": [dates[90]],
                "event_type": ["ai_disclosure"],
                "industry": ["银行"],
                "firm_name": ["测试"],
            }
        )
        from ashare_event.schema import load_events

        estimated = _estimate(load_events(events), prices, market, _cfg())
        self.assertTrue(bool(estimated.events["limit_hit"].iloc[0]))

    def test_peer_count_matches_calendar_window(self):
        frame = pd.DataFrame(
            {
                "event_id": ["a", "b", "c"],
                "industry": ["银行", "银行", "电子"],
                "event_date_aligned": pd.to_datetime(["2020-01-06", "2020-01-10", "2020-01-10"]),
            }
        )
        counted = add_peer_count(frame, days=5)
        self.assertListEqual(counted["peer_count_pm5"].tolist(), [1, 1, 0])

    def test_bh_does_not_shrink_pvalues(self):
        raw = np.array([0.01, 0.04, 0.03])
        adjusted = bh_adjust(raw)
        self.assertTrue(np.all(adjusted + 1e-12 >= raw))
        self.assertTrue(np.all(np.diff(np.sort(adjusted)) >= -1e-12))

    def test_audit_lists_actionable_rows(self):
        frame = pd.DataFrame(
            {
                "code": ["ABC", "999999", "600519", ""],
                "event_date": ["2024-03-18", "2024-03-18", "2024-03-16", None],
                "firm_name": ["", "", "", ""],
            }
        )
        report = audit_grid(frame)
        self.assertIn("错误", set(report["级别"]))
        self.assertIn("警告", set(report["级别"]))
        self.assertIn("提示", set(report["级别"]))
        self.assertTrue(report["问题"].str.contains("无法解析").any())

    def test_run_does_not_mutate_config_and_export_is_isolated(self):
        events, prices, market, _dates = synthetic(n_firms=6, n_days=220, shocks=[0.01] * 6, event_locs=[140] * 6)
        cfg = _cfg()
        cfg.mode = "local"
        before_windows = list(cfg.event_windows)
        before_regression = tuple(cfg.regression_window)
        with tempfile.TemporaryDirectory() as tmp:
            cfg.root = Path(tmp)
            result = run_study(
                cfg,
                regression_window=(-1, 1),
                run_placebo=False,
                export=True,
                output_dir=Path(tmp) / "isolated",
                events_frame=events,
                prices_frame=prices,
                market_frame=market,
            )
            self.assertEqual(list(cfg.event_windows), before_windows)
            self.assertEqual(tuple(cfg.regression_window), before_regression)
            self.assertEqual(tuple(result.config.regression_window), (-1, 1))
            manifest = Path(tmp) / "isolated" / "logs" / "manifest.json"
            self.assertTrue(manifest.exists())
            self.assertIn("events_hash", manifest.read_text(encoding="utf-8"))
            self.assertFalse((Path(tmp) / "paper" / "tabs" / "table_car.tex").exists())
            self.assertIn("不是", result.methods_zh)

    def test_industry_benchmark_matches_market_when_the_index_is_the_same(self):
        events, prices, market, _dates = synthetic(n_firms=4, shocks=[0.02] * 4, eps_sd=0.0, event_locs=[150, 152, 154, 156])
        events["industry"] = "电子"
        industry = market.copy()
        industry["industry"] = "电子"
        cfg = _cfg()
        market_fit = _estimate(events, prices, market, cfg, model="market")
        industry_fit = _estimate(events, prices, market, cfg, model="industry", industry_frame=industry)
        left = market_fit.events.set_index("event_id")["car_p0_p0"]
        right = industry_fit.events.set_index("event_id")["car_p0_p0"]
        np.testing.assert_allclose(right.reindex(left.index), left, atol=1e-8)

    def test_three_factor_recovers_a_known_shock(self):
        rng = np.random.default_rng(0)
        n_days = 160
        dates = pd.bdate_range("2021-01-04", periods=n_days)
        mkt = rng.normal(0, 0.01, n_days)
        smb = rng.normal(0, 0.005, n_days)
        hml = rng.normal(0, 0.005, n_days)
        mkt[0] = smb[0] = hml[0] = 0.0
        rets = 0.1 * mkt + 0.2 * smb - 0.3 * hml
        rets[0] = 0.0
        rets[100] += 0.02
        prices = pd.DataFrame({"code": "000001", "date": dates, "close": 10 * np.cumprod(1.0 + rets)})
        market = pd.DataFrame({"date": dates, "close": 1000 * np.cumprod(1.0 + mkt)})
        factors = pd.DataFrame({"date": dates, "mkt": mkt, "smb": smb, "hml": hml})
        from ashare_event.schema import load_events

        events = load_events(
            pd.DataFrame(
                {
                    "firm_id": ["F1"],
                    "code": ["000001"],
                    "event_date": [dates[100]],
                    "event_type": ["ai_disclosure"],
                    "industry": ["银行"],
                    "firm_name": ["测试"],
                }
            )
        )
        cfg = _cfg()
        cfg.model = "ff3"
        estimated = _estimate(events, prices, market, cfg, model="ff3", factors_frame=factors)
        self.assertAlmostEqual(float(estimated.events["beta"].iloc[0]), 0.1, places=3)
        self.assertAlmostEqual(float(estimated.events["beta_smb"].iloc[0]), 0.2, places=3)
        self.assertAlmostEqual(float(estimated.events["beta_hml"].iloc[0]), -0.3, places=3)
        self.assertAlmostEqual(float(estimated.events["car_p0_p0"].iloc[0]), 0.02, places=3)

    def test_vif_and_two_way_and_methods_language(self):
        signal = np.linspace(0.1, 0.9, 24)
        events, prices, market, _dates = synthetic(
            n_firms=24,
            shocks=(0.05 * signal).tolist(),
            betas=[1.0] * 24,
            eps_sd=0.004,
            seed=11,
            event_locs=[150 + (i % 5) for i in range(24)],
        )
        cfg = _cfg()
        kept = _estimate(events, prices, market, cfg).events
        notes: list[str] = []
        models = fit_car_regressions(
            kept,
            regression_window=(0, 0),
            covariates=["signal_cost"],
            cluster="firm_id",
            return_scale=100,
            notes=notes,
            two_way=True,
            wild_draws=49,
            rng=np.random.default_rng(1),
        )
        self.assertTrue(any(model.get("two_way") for model in models))
        wild = next(model["wild_p"]["signal_cost"] for model in models if model.get("wild_p"))
        self.assertGreaterEqual(wild, 0.0)
        self.assertLess(wild, 0.2)
        vif = variance_inflation(kept, ["signal_cost", "size_log"])
        self.assertIn("vif", vif.columns)
        zh, en = render_methods(cfg, n_events=24, n_firms=24, preferred="bmp", r_bar=0.0, simulated=False)
        self.assertIn("不是", zh)
        self.assertIn("not an average treatment effect", en)
        self.assertTrue(notes)

    def test_placebo_draws_stay_fast(self):
        events, prices, market, _dates = synthetic(
            n_firms=30,
            n_days=200,
            shocks=[0.01] * 30,
            eps_sd=0.01,
            event_locs=[120 + (i % 4) for i in range(30)],
        )
        cfg = _cfg()
        cfg.placebo_draws = 12
        cfg.placebo_modes = ["in_time", "in_space"]
        kept = _estimate(events, prices, market, cfg).events
        started = time.perf_counter()
        placebos = run_placebos(kept, prices, market, cfg, float(kept["car_p0_p0"].mean()), np.random.default_rng(2))
        self.assertLess(time.perf_counter() - started, 20.0)
        self.assertGreater(placebos["in_time"]["draws"], 5)
        self.assertIn("不是", placebos["in_time"]["sentence"])
        self.assertTrue(np.isfinite(placebos["in_space"]["percentile"]))


if __name__ == "__main__":
    unittest.main()
