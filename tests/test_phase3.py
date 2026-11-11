"""Phase 3: filing-text import, spec log, and background job status."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare_event.bundle import build_zip
from ashare_event.config import load_config, validate_windows
from ashare_event.jobs import (
    BACKGROUND_EVENT_THRESHOLD,
    execute_job,
    job_status,
    load_job_result,
    prepare_job,
    should_background,
    submit_job,
)
from ashare_event.methods_text import render_methods
from ashare_event.pipeline import run_study
from ashare_event.schema import load_events
from ashare_event.specs import spec_row, spec_sentence, spec_table
from ashare_event.text_import import events_from_filing_text, load_filing_events, locate_filing_panel, read_filing_table
from test_core import synthetic


def _cfg():
    cfg = load_config(ROOT / "config" / "study.yml")
    cfg.mode = "local"
    cfg.model = "market"
    cfg.estimation_window = (-80, -16)
    cfg.event_windows = [(0, 0), (-1, 1)]
    cfg.path_window = (-5, 5)
    cfg.regression_window = (-1, 1)
    cfg.min_estimation_obs = 40
    cfg.placebo_draws = 0
    cfg.wild_draws = 0
    cfg.verbose = False
    cfg.winsorize = 0
    cfg.market_lags = 0
    validate_windows(cfg)
    return cfg


class TextImportTests(unittest.TestCase):
    def test_factor_table_uses_available_date_without_a_second_shift(self):
        frame = pd.DataFrame(
            {
                "stkcd": [2230],
                "ticker": ["002230.SZ"],
                "available_date": ["2024-04-30"],
                "fiscal_year": [2023],
                "market": ["CN"],
                "firm_name": ["科大讯飞"],
                "mda_digital_per_k": [2.25],
                "mda_esg_per_k": [5.18],
                "mda_geopol_per_k": [0.4],
            }
        )
        out = events_from_filing_text(frame)
        self.assertEqual(out.loc[0, "code"], "002230")
        self.assertEqual(pd.Timestamp(out.loc[0, "event_date"]), pd.Timestamp("2024-04-30"))
        self.assertFalse(bool(out.loc[0, "after_close"]))
        self.assertAlmostEqual(float(out.loc[0, "signal_cost"]), 2.25)
        self.assertAlmostEqual(float(out.loc[0, "mda_esg_per_k"]), 5.18)
        self.assertAlmostEqual(float(out.loc[0, "mda_geopol_per_k"]), 0.4)
        self.assertNotIn("legitimacy_gap", out.columns)
        self.assertEqual(out.attrs["signal_source"], "mda_digital_per_k")
        self.assertIn("年报文本特征引擎", out.attrs["import_note"])
        self.assertNotEqual(pd.Timestamp(out.loc[0, "event_date"]), pd.Timestamp("2023-12-31"))

    def test_announce_date_is_the_event_and_after_close_is_on(self):
        frame = pd.DataFrame(
            {
                "ticker": ["000001.SZ", "600000.SH", "000002.SZ"],
                "announce_date": ["2024-04-12", "2024-04-10", "2024-04-01"],
                "available_date": ["2024-04-13", "2024-04-12", "2024-04-03"],
                "filing_date": ["2024-04-12", "2024-04-10", "2024-04-01"],
                "mda_digital_per_k": [1.5, 0.0, 4.0],
                "firm_name": ["平安银行", "浦发银行", "万科"],
                "market": ["CN", "US", "CN"],
            }
        )
        out = events_from_filing_text(frame)
        self.assertEqual(len(out), 2)
        first = out.loc[out["code"].eq("000001")].iloc[0]
        second = out.loc[out["code"].eq("000002")].iloc[0]
        self.assertEqual(pd.Timestamp(first["event_date"]), pd.Timestamp("2024-04-12"))
        self.assertTrue(bool(first["after_close"]))
        self.assertAlmostEqual(float(first["signal_cost"]), 1.5)
        self.assertEqual(pd.Timestamp(second["event_date"]), pd.Timestamp("2024-04-03"))
        self.assertFalse(bool(second["after_close"]))
        self.assertIn("非 A 股", out.attrs["import_note"])

    def test_missing_frequency_stays_missing(self):
        absent = events_from_filing_text(
            pd.DataFrame({"stkcd": ["000001"], "announce_date": ["2024-04-12"], "firm_name": ["平安银行"]})
        )
        self.assertNotIn("signal_cost", absent.columns)
        present = events_from_filing_text(
            pd.DataFrame(
                {
                    "code": ["000001", "000002"],
                    "announce_date": ["2024-04-12", "2024-04-13"],
                    "mda_digital_per_k": [None, 0],
                }
            )
        )
        self.assertTrue(pd.isna(present.loc[0, "signal_cost"]))
        self.assertEqual(float(present.loc[1, "signal_cost"]), 0.0)
        loaded = load_events(present)
        self.assertTrue(pd.isna(loaded.loc[loaded["code"].eq("000001"), "signal_cost"]).all())
        self.assertIn("没有记成 0", present.attrs["import_note"])

    def test_long_panel_keeps_the_mda_frequency_and_the_ready_row(self):
        frame = pd.DataFrame(
            {
                "stkcd": ["000001", "000001", "000001", "000001"],
                "announce_date": ["2024-04-12", "2024-04-12", "2024-05-01", "2024-05-01"],
                "section": ["risk", "mda", "mda", "mda"],
                "digital_per_k": [9.0, 3.0, 1.0, 8.0],
                "esg_per_k": [99.0, 1.5, 2.0, 4.0],
                "pit_ready": [1, 0, 0, 1],
                "firm_name": ["风险行", "讨论行", "低", "高"],
            }
        )
        out = events_from_filing_text(frame)
        self.assertEqual(len(out), 2)
        first = out.loc[out["event_date"].eq(pd.Timestamp("2024-04-12"))].iloc[0]
        second = out.loc[out["event_date"].eq(pd.Timestamp("2024-05-01"))].iloc[0]
        self.assertAlmostEqual(float(first["signal_cost"]), 3.0)
        self.assertAlmostEqual(float(first["mda_esg_per_k"]), 1.5)
        self.assertEqual(first["firm_name"], "风险行")
        self.assertAlmostEqual(float(second["signal_cost"]), 8.0)
        self.assertEqual(second["firm_name"], "高")
        self.assertTrue(bool(first["after_close"]))

    def test_csv_roundtrip_reads_the_same_dates(self):
        frame = pd.DataFrame(
            {
                "filing_date": ["2024-04-12"],
                "available_date": ["2024-04-13"],
                "stkcd": ["688111"],
                "mda_digital_per_k": [4],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "panel_factor.csv"
            frame.to_csv(path, index=False)
            out = events_from_filing_text(read_filing_table(path))
        self.assertEqual(pd.Timestamp(out.loc[0, "event_date"]), pd.Timestamp("2024-04-12"))
        self.assertTrue(bool(out.loc[0, "after_close"]))
        self.assertEqual(out.loc[0, "event_type"], "annual_report")


class SpecLogTests(unittest.TestCase):
    def test_two_rows_do_not_change_car_and_the_zip_lists_them(self):
        events, prices, market, _dates = synthetic(
            n_firms=4,
            n_days=220,
            shocks=[0.02] * 4,
            event_locs=[140] * 4,
        )
        result = run_study(
            _cfg(),
            regression_window=(-1, 1),
            run_placebo=False,
            export=False,
            events_frame=events,
            prices_frame=prices,
            market_frame=market,
        )
        self.assertGreater(len(result.events), 0)
        means = result.tests["mean"].tolist()
        self.assertEqual(len(result.specs), 1)
        self.assertIn("共试了 1 种规格", spec_sentence(result.specs))
        result.specs = spec_table(
            [
                spec_row(result, source="演示样本", industries=None, event_types=None),
                spec_row(result, source="我的事件表", industries=["银行"], event_types=None),
            ]
        )
        self.assertEqual(result.tests["mean"].tolist(), means)
        sentence = spec_sentence(result.specs)
        self.assertIn("共试了 2 种规格", sentence)
        self.assertIn("没有按这", sentence)
        self.assertIn("Holm", sentence)
        with zipfile.ZipFile(io.BytesIO(build_zip(result))) as archive:
            self.assertIn("csv/specs_tried.csv", archive.namelist())
            self.assertIn("tex/table_specs.tex", archive.namelist())
            note = archive.read("methods_note.txt").decode("utf-8")
            tex = archive.read("tex/table_specs.tex").decode("utf-8")
        self.assertIn("共试了 2 种规格", note)
        self.assertIn("共试了 2 种规格", tex)
        self.assertNotIn("***", tex.split("tablenotes")[0])

    def test_signal_proxy_is_named_only_when_the_import_set_it(self):
        flagged = _cfg()
        flagged.extra["signal_source"] = "mda_digital_per_k"
        zh, en = render_methods(flagged, n_events=4, n_firms=4, preferred="bmp", r_bar=0.0, simulated=False)
        plain, _en = render_methods(_cfg(), n_events=4, n_firms=4, preferred="bmp", r_bar=0.0, simulated=False)
        self.assertIn("每千词", zh)
        self.assertIn("年报文本特征引擎", zh)
        self.assertIn("annual-report text engine", en)
        self.assertNotIn("每千词", plain)


class JobTests(unittest.TestCase):
    def setUp(self):
        self._old_job_dir = os.environ.pop("ASHARE_JOB_DIR", None)

    def tearDown(self):
        if self._old_job_dir is not None:
            os.environ["ASHARE_JOB_DIR"] = self._old_job_dir

    def test_worker_writes_status_without_a_live_thread(self):
        def worker(progress):
            progress(1, 2, "一步")
            progress(2, 2, "完成")
            return {"n": 3}

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            job_id = prepare_job(base, {"source": "测试"})
            execute_job(job_id, worker, base=base)
            status = job_status(job_id, base)
            self.assertEqual(status["state"], "complete")
            self.assertEqual(status["done"], 2)
            self.assertEqual(status["label"], "计算完成")
            self.assertEqual(load_job_result(job_id, base), {"n": 3})

    def test_a_failed_worker_does_not_pretend_to_finish(self):
        def worker(progress):
            progress(0, 1, "下载")
            raise RuntimeError("行情没有下完")

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            job_id = prepare_job(base)
            execute_job(job_id, worker, base=base)
            status = job_status(job_id, base)
            self.assertEqual(status["state"], "error")
            self.assertIn("行情没有下完", status["message"])
            self.assertIsNone(load_job_result(job_id, base))

    def test_a_prepared_job_without_a_thread_is_lost(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_id = prepare_job(Path(tmp))
            status = job_status(job_id, Path(tmp))
        self.assertEqual(status["state"], "lost")
        self.assertIn("后台线程", status["message"])

    def test_submit_finishes_and_can_be_read_back(self):
        def worker(progress):
            progress(1, 1, "完成")
            return {"ok": True}

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            job_id = submit_job(worker, base=base, meta={"source": "测试"})
            status = {"state": "running"}
            for _ in range(50):
                status = job_status(job_id, base)
                if status["state"] in {"complete", "error", "lost"}:
                    break
                time.sleep(0.05)
            self.assertEqual(status["state"], "complete")
            self.assertEqual(load_job_result(job_id, base), {"ok": True})

    def test_background_threshold_leaves_demo_and_small_runs_on_the_page(self):
        self.assertFalse(
            should_background(n_events=BACKGROUND_EVENT_THRESHOLD, mode="akshare", source="演示样本", forced=False)
        )
        self.assertFalse(should_background(n_events=29, mode="akshare", source="我的事件表", forced=False))
        self.assertTrue(should_background(n_events=30, mode="akshare", source="对照案例", forced=False))
        self.assertTrue(should_background(n_events=30, mode="akshare", source="年报词频", forced=False))
        self.assertTrue(should_background(n_events=1, mode="demo", source="演示样本", forced=True))
        self.assertFalse(should_background(n_events=80, mode="local", source="我的事件表", forced=False))

    def test_locate_reads_the_text_engine_export_in_place(self):
        previous = os.environ.get("ASHARE_FILING_TEXT_DIR")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                wanted = root / "data" / "factor_panel" / "output"
                decoy = root / "data" / "output"
                wanted.mkdir(parents=True)
                decoy.mkdir(parents=True)
                pd.DataFrame(
                    {
                        "stkcd": ["2230"],
                        "available_date": ["2024-04-30"],
                        "fiscal_year": [2023],
                        "mda_digital_per_k": [2.25],
                        "mda_esg_per_k": [1.5],
                        "mda_geopol_per_k": [0.0],
                    }
                ).to_csv(wanted / "panel_factor.csv", index=False)
                pd.DataFrame({"stkcd": ["1"], "available_date": ["2020-01-01"]}).to_csv(
                    decoy / "panel_factor.csv", index=False
                )
                wide_only = root / "data" / "output"
                wide_only.mkdir(parents=True, exist_ok=True)
                (wide_only / "panel_wide.parquet").write_bytes(b"not-a-panel")
                os.environ["ASHARE_FILING_TEXT_DIR"] = str(root)
                found = locate_filing_panel()
                self.assertEqual(found, wanted / "panel_factor.csv")
                events = load_filing_events(found)
                self.assertEqual(events.loc[0, "code"], "002230")
                self.assertEqual(int(events.loc[0, "fiscal_year"]), 2023)
                self.assertAlmostEqual(float(events.loc[0, "signal_cost"]), 2.25)
                self.assertNotIn("legitimacy_gap", events.columns)
                (wanted / "panel_factor.csv").unlink()
                (decoy / "panel_factor.csv").unlink()
                later = root / "data" / "validation" / "output"
                later.mkdir(parents=True)
                events.to_csv(later / "panel_factor.csv", index=False)
                os.environ["ASHARE_FILING_TEXT_DIR"] = str(root)
                self.assertEqual(locate_filing_panel(), later / "panel_factor.csv")
                os.environ["ASHARE_FILING_TEXT_DIR"] = str(later / "panel_factor.csv")
                self.assertEqual(locate_filing_panel(), later / "panel_factor.csv")
                os.environ["ASHARE_FILING_TEXT_DIR"] = str(root / "missing")
                self.assertIsNone(locate_filing_panel())
        finally:
            if previous is None:
                os.environ.pop("ASHARE_FILING_TEXT_DIR", None)
            else:
                os.environ["ASHARE_FILING_TEXT_DIR"] = previous


if __name__ == "__main__":
    unittest.main()
