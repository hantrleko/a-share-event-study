"""Tests for the event-study core. No network and no paper-ready claims."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare_event.config import ConfigError, load_config
from ashare_event.inference import kp_factor
from ashare_event.market import MarketDataError, load_or_download_stock, normalize_ohlc
from ashare_event.pipeline import run_study
from ashare_event.returns import estimate_events
from ashare_event.schema import SchemaError, apply_column_roles, load_events, normalize_code
from ashare_event.tables import car_table
from ashare_event.theory import render_theory


def synthetic(
    *,
    n_days=260,
    n_firms=1,
    event_locs=None,
    shocks=None,
    betas=None,
    eps_sd=0.0,
    seed=7,
    factor_sd=0.0,
    drop_event_row=False,
    sunday_event=False,
):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-06", periods=n_days)
    if event_locs is None:
        event_locs = [180] * n_firms
    shocks = [0.03] * n_firms if shocks is None else shocks
    betas = [1.2] * n_firms if betas is None else betas
    market_ret = rng.normal(0.0002, 0.01, size=n_days)
    market_ret[0] = 0.0
    factor = rng.normal(0.0, factor_sd, size=n_days) if factor_sd else np.zeros(n_days)
    frames = []
    rows = []
    for i in range(n_firms):
        eps = rng.normal(0.0, eps_sd, size=n_days) if eps_sd else np.zeros(n_days)
        rets = betas[i] * market_ret + factor + eps
        rets[0] = 0.0
        rets[event_locs[i]] += shocks[i]
        code = f"{i + 1:06d}"
        frame = pd.DataFrame({"code": code, "date": dates, "close": 10 * np.cumprod(1.0 + rets)})
        if drop_event_row and i == 0:
            frame = frame.loc[frame["date"] != dates[event_locs[i]]].copy()
        frames.append(frame)
        event_date = dates[event_locs[i]]
        if sunday_event and i == 0:
            event_date = event_date - pd.Timedelta(days=1)
        rows.append(
            {
                "firm_id": f"F{i+1}",
                "code": code,
                "event_date": event_date,
                "event_type": "ai_disclosure",
                "industry": "电子" if i % 2 == 0 else "银行",
                "signal_cost": shocks[i] / 0.05 if shocks[i] else 0.0,
                "size_log": 15 + i / 10,
                "legitimacy_gap": 0.2,
            }
        )
    events = load_events(pd.DataFrame(rows))
    market = pd.DataFrame({"date": dates, "close": 1000 * np.cumprod(1.0 + market_ret)})
    return events, pd.concat(frames, ignore_index=True), market, dates


def _estimate(events, prices, market, windows=((0, 0), (-1, 1)), estimation=(-100, -21)):
    return estimate_events(
        events,
        prices,
        market,
        estimation_window=estimation,
        event_windows=list(windows),
        path_window=(-10, 10),
        regression_window=(0, 0),
        model="market",
        min_estimation_obs=40,
    )


class DesignTests(unittest.TestCase):
    def test_exact_shock_and_beta(self):
        events, prices, market, dates = synthetic(betas=[1.2], shocks=[0.03], eps_sd=0.0)
        monday = dates[180]
        self.assertEqual(monday.dayofweek, monday.dayofweek)
        estimated = _estimate(events, prices, market)
        self.assertEqual(len(estimated.events), 1)
        self.assertAlmostEqual(estimated.events["beta"].iloc[0], 1.2, places=6)
        self.assertAlmostEqual(estimated.events["car_p0_p0"].iloc[0], 0.03, places=6)
        path = estimated.panel.dropna(subset=["car_path"])
        before = path.loc[path["rel_day"] == -1, "car_path"].iloc[0]
        on_day = path.loc[path["rel_day"] == 0, "car_path"].iloc[0]
        self.assertAlmostEqual(before, 0.0, places=6)
        self.assertAlmostEqual(on_day - before, 0.03, places=6)

    def test_sunday_rolls_to_monday_shock(self):
        dates = pd.bdate_range("2020-01-06", periods=260)
        mondays = np.where(dates.dayofweek == 0)[0]
        loc = int(mondays[mondays > 120][0])
        events, prices, market, _dates = synthetic(event_locs=[loc], shocks=[0.04], sunday_event=True)
        estimated = _estimate(events, prices, market)
        self.assertTrue(bool(estimated.events["rolled_forward"].iloc[0]))
        self.assertEqual(pd.Timestamp(estimated.events["event_date_aligned"].iloc[0]), dates[loc])
        self.assertAlmostEqual(estimated.events["car_p0_p0"].iloc[0], 0.04, places=6)

    def test_short_history_and_missing_day(self):
        events, prices, market, _dates = synthetic(event_locs=[8], n_days=80)
        estimated = _estimate(events, prices, market, estimation=(-60, -11))
        self.assertIn("short_history", set(estimated.dropped["reason"]))
        events, prices, market, _dates = synthetic(drop_event_row=True)
        estimated = _estimate(events, prices, market)
        self.assertIn("incomplete_window", set(estimated.dropped["reason"]))

    def test_duplicate_event_and_code(self):
        frame = pd.DataFrame({"code": ["1", "1"], "event_date": ["2020-01-06", "2020-01-06"]})
        with self.assertRaises(SchemaError):
            load_events(frame)
        self.assertEqual(normalize_code("600000.0"), "600000")
        self.assertEqual(normalize_code("SH600519"), "600519")

    def test_overlapping_window_is_rejected(self):
        cfg = load_config(ROOT / "config" / "study.yml")
        cfg.estimation_window = (-30, -5)
        cfg.event_windows = [(-5, 5)]
        from ashare_event.config import validate_windows

        with self.assertRaises(ConfigError):
            validate_windows(cfg)

    def test_kp_shrinks_and_correlation_is_detected(self):
        self.assertLess(kp_factor(0.1, 50), 1)
        self.assertAlmostEqual(kp_factor(0.0, 40), 1.0, places=6)
        events, prices, market, _dates = synthetic(n_firms=12, factor_sd=0.02, eps_sd=0.002, shocks=[0.0] * 12)
        estimated = _estimate(events, prices, market)
        from ashare_event.inference import mean_residual_correlation

        r_bar = mean_residual_correlation(estimated.panel, estimated.events["event_id"], 30)
        self.assertGreater(r_bar, 0.3)
        self.assertLess(kp_factor(r_bar, len(estimated.events)), 0.7)

    def test_placebo_and_regression_slope(self):
        signal = np.linspace(0.1, 0.9, 24)
        shocks = (0.05 * signal).tolist()
        events, prices, market, _dates = synthetic(
            n_firms=24,
            shocks=shocks,
            betas=[1.0] * 24,
            eps_sd=0.006,
            seed=11,
            event_locs=[150 + (i % 5) for i in range(24)],
        )
        events["year"] = 2021
        cfg = load_config(ROOT / "config" / "study.yml")
        cfg.estimation_window = (-80, -16)
        cfg.event_windows = [(0, 0), (-1, 1)]
        cfg.path_window = (-5, 5)
        cfg.regression_window = (0, 0)
        cfg.min_estimation_obs = 40
        cfg.placebo_draws = 25
        cfg.placebo_modes = ["in_time"]
        cfg.placebo_offsets = (-40, -20)
        cfg.verbose = False
        from ashare_event.config import validate_windows
        from ashare_event.placebo import run_placebos
        from ashare_event.regression import fit_car_regressions

        validate_windows(cfg)
        estimated = estimate_events(
            events,
            prices,
            market,
            estimation_window=cfg.estimation_window,
            event_windows=cfg.event_windows,
            path_window=cfg.path_window,
            regression_window=cfg.regression_window,
            model="market",
            min_estimation_obs=40,
        )
        kept = estimated.events
        error = (kept["car_p0_p0"] - kept["signal_cost"] * 0.05).mean()
        self.assertLess(abs(error), 0.01)
        models = fit_car_regressions(
            kept,
            regression_window=(0, 0),
            covariates=["signal_cost"],
            cluster="firm_id",
            return_scale=100,
        )
        slope = models[1]["b"]["signal_cost"]
        self.assertGreater(slope, 3.0)
        self.assertLess(slope, 7.0)
        self.assertEqual(models[0]["engine"], "pyfixest")
        self.assertEqual(models[0]["g"], 24)
        actual = float(kept["car_p0_p0"].mean())
        placebo = run_placebos(kept, prices, market, cfg, actual, np.random.default_rng(3))
        self.assertLess(placebo["in_time"]["p_value"], 0.05)
        self.assertGreater(placebo["in_time"]["draws"], 10)

    def test_table_and_theory_and_demo_export(self):
        events, prices, market, _dates = synthetic(n_firms=6, shocks=[0.02] * 6, eps_sd=0.01)
        estimated = _estimate(events, prices, market)
        from ashare_event.inference import summarize_window

        row = summarize_window(
            estimated.events,
            estimated.panel,
            (0, 0),
            "p0_p0",
            estimation_window=(-100, -21),
            model="market",
            r_bar=0.0,
            min_share=0.5,
        )
        tex = car_table(pd.DataFrame([row]), "zh", 100, True, "firm_id")
        self.assertIn("\\toprule", tex)
        self.assertIn("threeparttable", tex)
        self.assertIn("模拟数据", tex)
        self.assertNotIn("\\begin{tabular}{|", tex)
        note = render_theory(["geopolitics"], ["signal_cost"], "zh")
        self.assertIn("候选", note)
        self.assertIn("现金流", note)
        self.assertIn("不能", note)
        self.assertIn("signal_cost", note)
        cfg = load_config(ROOT / "config" / "study.yml")
        cfg.mode = "demo"
        cfg.verbose = False
        cfg.placebo_draws = 8
        cfg.placebo_modes = ["in_time"]
        with tempfile.TemporaryDirectory() as tmp:
            cfg.root = Path(tmp)
            result = run_study(cfg, event_types=["ai_disclosure"], run_placebo=True, export=True)
            self.assertTrue(result.simulated)
            self.assertTrue((Path(tmp) / "paper" / "tabs" / "table_car.tex").exists())
            self.assertTrue((Path(tmp) / "paper" / "figs" / "fig_car_path.pdf").read_bytes().startswith(b"%PDF"))
            self.assertIn("SIMULATED", (Path(tmp) / "data" / "processed" / "SIMULATED.txt").read_text(encoding="utf-8"))
            self.assertGreater(result.events["car_m1_p1"].mean(), 0)
            regression = (Path(tmp) / "paper" / "tabs" / "table_regression.tex").read_text(encoding="utf-8")
            self.assertIn("\\toprule", regression)
            self.assertIn("信号成本", regression)
            banks = run_study(cfg, industries=["银行"], run_placebo=False, export=False)
            self.assertTrue((banks.events["industry"] == "银行").all())
            self.assertLess(len(banks.events), len(result.events) + 40)

    def test_user_column_names_map_onto_event_fields(self):
        frame = pd.DataFrame(
            {
                "证券代码": ["1", "600519"],
                "公告日期": ["2024-03-18", "2024-05-09"],
                "类型": ["AI披露", "ESG"],
                "所属行业": ["银行", "食品饮料"],
                "信号成本": ["0.8", "0.2"],
            }
        )
        mapped = apply_column_roles(
            frame,
            {
                "code": "证券代码",
                "event_date": "公告日期",
                "event_type": "类型",
                "industry": "所属行业",
            },
        )
        self.assertEqual(mapped["code"].iloc[0], "000001")
        self.assertEqual(set(mapped["event_type"]), {"ai_disclosure", "esg"})
        self.assertIn("signal_cost", mapped.columns)
        self.assertAlmostEqual(float(mapped["signal_cost"].iloc[0]), 0.8)
        from ashare_event.schema import guess_roles, rename_to_standard

        roles = guess_roles(list(frame.columns))
        self.assertEqual(roles["code"], "证券代码")
        self.assertEqual(roles["event_date"], "公告日期")
        renamed = rename_to_standard(frame)
        self.assertIn("signal_cost", renamed.columns)

    def test_ohlc_normalizer_and_cache_guard(self):
        frame = pd.DataFrame({"日期": ["2024-01-02", "2024-01-03"], "收盘": [10.0, 10.5]})
        out = normalize_ohlc(frame, code="000001")
        self.assertListEqual(list(out.columns), ["code", "date", "close"])
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            path = raw / "prices" / "000001.parquet"
            path.parent.mkdir(parents=True)
            cached = pd.DataFrame(
                {"code": ["000001"], "date": [pd.Timestamp("2024-01-02")], "close": [10.0]}
            )
            cached.to_parquet(path, index=False)
            path.with_suffix(".json").write_text(
                '{"start": "2024-01-02", "end": "2024-01-03", "adjust": "qfq"}',
                encoding="utf-8",
            )
            with self.assertRaises(MarketDataError):
                load_or_download_stock("000001", "2020-01-01", "2024-12-31", "qfq", raw)


if __name__ == "__main__":
    unittest.main()
