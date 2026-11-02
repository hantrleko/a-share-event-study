"""V2 checks. Extra columns and a separate robustness grid. No wild bootstrap."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare_event.bundle import build_zip
from ashare_event.config import clone_config
from ashare_event.inference import summarize_window
from ashare_event.pipeline import run_study
from ashare_event.returns import estimate_events, winsorize_events
from ashare_event.robustness import run_robustness
from test_core import synthetic
from test_v1 import _cfg


def _fit(events, prices, market, **kwargs):
    cfg = _cfg()
    return estimate_events(
        events,
        prices,
        market,
        estimation_window=kwargs.pop("estimation_window", cfg.estimation_window),
        event_windows=kwargs.pop("event_windows", [(0, 0)]),
        path_window=kwargs.pop("path_window", (-10, 10)),
        regression_window=kwargs.pop("regression_window", (0, 0)),
        model=kwargs.pop("model", "market"),
        min_estimation_obs=kwargs.pop("min_estimation_obs", 40),
        **kwargs,
    )


class VersionTwoTests(unittest.TestCase):
    def test_one_day_bhar_matches_car_and_lags_zero_is_unchanged(self):
        events, prices, market, _dates = synthetic(betas=[1.2], shocks=[0.03], eps_sd=0.0)
        plain = _fit(events, prices, market, market_lags=0)
        default = _fit(events, prices, market)
        lagged = _fit(events, prices, market, market_lags=1)
        car = float(plain.events["car_p0_p0"].iloc[0])
        self.assertAlmostEqual(car, 0.03, places=6)
        self.assertAlmostEqual(float(plain.events["beta"].iloc[0]), 1.2, places=6)
        self.assertAlmostEqual(float(plain.events["bhar_p0_p0"].iloc[0]), car, places=8)
        np.testing.assert_allclose(default.events["car_p0_p0"], plain.events["car_p0_p0"], atol=1e-12)
        self.assertAlmostEqual(float(lagged.events["car_p0_p0"].iloc[0]), 0.03, places=4)
        self.assertAlmostEqual(float(lagged.events["beta"].iloc[0]), 1.2, places=4)

    def test_winsorize_is_off_by_default_and_clips_when_asked(self):
        shocks = [0.01 + i * 0.01 for i in range(12)]
        events, prices, market, _dates = synthetic(
            n_firms=12,
            shocks=shocks,
            eps_sd=0.0,
            event_locs=[150 + (i % 4) for i in range(12)],
        )
        estimated = _fit(events, prices, market)
        original = estimated.events["car_p0_p0"].to_numpy(dtype=float)
        untouched = winsorize_events(estimated.events, 0.0)
        np.testing.assert_allclose(untouched["car_p0_p0"], original, atol=1e-12)
        clipped = winsorize_events(estimated.events, 0.2)
        self.assertLess(float(np.nanmax(clipped["car_p0_p0"])), float(np.nanmax(original)))

    def test_sign_calendar_and_anticipation_stay_out_of_the_main_table(self):
        events, prices, market, _dates = synthetic(
            n_firms=10,
            shocks=[0.05] * 10,
            eps_sd=0.002,
            event_locs=[150] * 10,
            seed=19,
        )
        estimated = _fit(events, prices, market, event_windows=[(0, 0), (-1, 1)])
        same_day = summarize_window(
            estimated.events,
            estimated.panel,
            (0, 0),
            "p0_p0",
            estimation_window=(-80, -16),
            model="market",
            r_bar=0.0,
            min_share=0.5,
        )
        spread = summarize_window(
            estimated.events,
            estimated.panel,
            (-1, 1),
            "m1_p1",
            estimation_window=(-80, -16),
            model="market",
            r_bar=0.0,
            min_share=0.5,
        )
        self.assertTrue(np.isnan(same_day["t_calendar"]))
        self.assertTrue(np.isfinite(spread["t_calendar"]))
        self.assertTrue(np.isfinite(spread["t_sign"]))
        self.assertGreater(spread["t_sign"], 0)
        self.assertIn(spread["preferred"], {"bmp", "kp", "portfolio"})

        cfg = _cfg()
        cfg.mode = "local"
        cfg.event_windows = [(0, 0)]
        cfg.regression_window = (0, 0)
        cfg.path_window = (-10, 10)
        cfg.anticipation_window = (-10, -6)
        cfg.winsorize = 0
        cfg.market_lags = 0
        held = list(cfg.event_windows)
        result = run_study(
            cfg,
            run_placebo=False,
            export=False,
            events_frame=events,
            prices_frame=prices,
            market_frame=market,
        )
        self.assertEqual(list(result.config.event_windows), held)
        self.assertEqual(len(result.events), 10)
        self.assertEqual(result.anticipation["t1"], -10)
        self.assertNotIn(-10, set(result.tests["t1"].astype(int)))
        self.assertTrue((result.tests["p_bh"].notna()).all())

    def test_factor_models_and_limit_span(self):
        rng = np.random.default_rng(4)
        n_days = 220
        dates = pd.bdate_range("2020-01-06", periods=n_days)
        mkt = rng.normal(0.0, 0.01, size=n_days)
        smb = rng.normal(0.0, 0.01, size=n_days)
        hml = rng.normal(0.0, 0.01, size=n_days)
        rmw = rng.normal(0.0, 0.01, size=n_days)
        cma = rng.normal(0.0, 0.01, size=n_days)
        umd = rng.normal(0.0, 0.01, size=n_days)
        rets = 0.1 * mkt + 0.2 * smb - 0.3 * hml + 0.4 * rmw - 0.15 * cma
        rets[0] = 0.0
        rets[180] += 0.02
        events = synthetic(n_days=n_days, shocks=[0.0])[0].iloc[:1].copy()
        events["event_date"] = dates[180]
        events["code"] = "000001"
        prices = pd.DataFrame({"code": "000001", "date": dates, "close": 10 * np.cumprod(1.0 + rets)})
        market = pd.DataFrame({"date": dates, "close": 1000 * np.cumprod(1.0 + np.concatenate([[0.0], mkt[1:]]))})
        factors = pd.DataFrame({"date": dates, "mkt": mkt, "smb": smb, "hml": hml, "rmw": rmw, "cma": cma, "umd": umd})
        fitted = _fit(events, prices, market, model="ff5", factors_frame=factors)
        row = fitted.events.iloc[0]
        self.assertAlmostEqual(float(row["beta"]), 0.1, places=3)
        self.assertAlmostEqual(float(row["beta_rmw"]), 0.4, places=3)
        self.assertAlmostEqual(float(row["beta_cma"]), -0.15, places=3)
        self.assertAlmostEqual(float(row["car_p0_p0"]), 0.02, places=3)
        carhart_rets = 0.1 * mkt + 0.2 * smb - 0.3 * hml + 0.25 * umd
        carhart_rets[0] = 0.0
        carhart_rets[180] += 0.02
        carhart_prices = pd.DataFrame({"code": "000001", "date": dates, "close": 10 * np.cumprod(1.0 + carhart_rets)})
        carhart = _fit(events, carhart_prices, market, model="carhart", factors_frame=factors)
        self.assertAlmostEqual(float(carhart.events["beta_umd"].iloc[0]), 0.25, places=3)

        limited = np.full(n_days, 0.001)
        limited[0] = 0.0
        limited[180] = 0.099
        limited[181] = 0.099
        limit_prices = pd.DataFrame({"code": "000001", "date": dates, "close": 10 * np.cumprod(1.0 + limited)})
        quiet = pd.DataFrame({"date": dates, "close": 1000 * np.cumprod(1.0 + np.full(n_days, 0.0002))})
        quiet_events = events.copy()
        span = _fit(
            quiet_events,
            limit_prices,
            quiet,
            event_windows=[(0, 0), (0, 1)],
            market_lags=0,
        )
        self.assertTrue(bool(span.events["limit_hit"].iloc[0]))
        self.assertGreaterEqual(int(span.events["limit_hit_days"].iloc[0]), 2)

    def test_robustness_replication_and_clone(self):
        events, prices, market, _dates = synthetic(
            n_firms=8,
            shocks=[0.02] * 8,
            eps_sd=0.004,
            event_locs=[150 + i for i in range(8)],
        )
        cfg = _cfg()
        cfg.mode = "local"
        cfg.event_windows = [(0, 0)]
        cfg.regression_window = (0, 0)
        cfg.path_window = (-10, 10)
        cfg.anticipation_window = None
        cfg.winsorize = 0.01
        cfg.market_lags = 1
        cloned = clone_config(cfg)
        cfg.winsorize = 0
        cfg.market_lags = 0
        self.assertEqual(cloned.winsorize, 0.01)
        self.assertEqual(cloned.market_lags, 1)
        table = run_robustness(
            cloned,
            events_frame=events,
            prices_frame=prices,
            market_frame=market,
        )
        self.assertGreaterEqual(len(table), 2)
        self.assertIn("mean_car", table.columns)
        self.assertIn("mean_bhar", table.columns)
        with tempfile.TemporaryDirectory() as tmp:
            export_cfg = _cfg()
            export_cfg.mode = "local"
            export_cfg.event_windows = [(0, 0)]
            export_cfg.regression_window = (0, 0)
            export_cfg.path_window = (-10, 10)
            export_cfg.anticipation_window = (-10, -6)
            result = run_study(
                export_cfg,
                run_placebo=False,
                export=True,
                output_dir=tmp,
                events_frame=events,
                prices_frame=prices,
                market_frame=market,
            )
            root = Path(tmp)
            self.assertTrue((root / "logs" / "replication" / "study_repro.yml").exists())
            self.assertTrue((root / "logs" / "replication" / "run_replication.py").exists())
            self.assertTrue((root / "logs" / "replication" / "events.csv").exists())
            self.assertTrue((root / "tables" / "table_diagnostics.tex").exists())
            self.assertTrue((root / "logs" / "replication" / "event_car.dta").exists() or (root / "logs" / "replication" / "stata_note.txt").exists())
            payload = build_zip(result)
            self.assertIn(b"study_repro.yml", payload)
            text = (root / "tables" / "table_diagnostics.tex").read_text(encoding="utf-8")
            self.assertIn("BHAR", text)
            self.assertNotIn(str(root), (root / "logs" / "replication" / "study_repro.yml").read_text(encoding="utf-8"))
