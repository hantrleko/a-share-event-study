"""Built-in reference data. No network and no paper-ready claims."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare_event.pipeline import _sample_account
from ashare_event.reference import (
    asof_l1,
    ch_frame_from_table,
    class_l1_index,
    l1_name,
    load_case_events,
    normalize_membership,
    normalize_sw_catalog,
)
from ashare_event.returns import estimate_events


def _fit(events, prices, market, **kwargs):
    return estimate_events(
        events,
        prices,
        market,
        estimation_window=(-100, -21),
        event_windows=[(0, 0)],
        path_window=(-10, 10),
        regression_window=(0, 0),
        min_estimation_obs=40,
        **kwargs,
    )


class ReferenceTests(unittest.TestCase):
    def test_ch3_maps_vmg_into_hml_and_scales_percent(self):
        decimal = pd.DataFrame(
            {
                "date": [20000104, 20000105],
                "rf_dly": [0.0001, 0.0001],
                "mktrf": [0.01, -0.02],
                "SMB": [0.001, 0.002],
                "VMG": [-0.003, 0.004],
            }
        )
        out = ch_frame_from_table(decimal)
        self.assertAlmostEqual(float(out["mkt"].iloc[0]), 0.01)
        self.assertAlmostEqual(float(out["hml"].iloc[0]), -0.003)
        self.assertEqual(out["factor_source"].iloc[0], "ch3_vmg")
        percent = decimal.copy()
        percent[["rf_dly", "mktrf", "SMB", "VMG"]] = percent[["rf_dly", "mktrf", "SMB", "VMG"]] * 100
        scaled = ch_frame_from_table(percent)
        self.assertAlmostEqual(float(scaled["mkt"].iloc[0]), 0.01, places=6)

    def test_asof_ignores_a_later_reclassification(self):
        membership = pd.DataFrame(
            {
                "code": ["000001", "000001"],
                "start": pd.to_datetime(["2020-01-01", "2024-01-01"]),
                "industry_code": ["801780", "801790"],
                "industry_name": ["", ""],
            }
        )
        catalog = normalize_sw_catalog(
            pd.DataFrame({"指数代码": ["801780", "801790"], "指数名称": ["银行", "非银金融"]})
        )
        self.assertEqual(asof_l1(membership, catalog, "000001", "2023-07-13"), "银行")
        self.assertEqual(asof_l1(membership, catalog, "000001", "2024-06-01"), "非银金融")
        self.assertEqual(class_l1_index("480301"), "801780")
        self.assertEqual(class_l1_index("480301.0"), "801780")
        self.assertIsNone(class_l1_index("850111"))
        self.assertEqual(l1_name("480301", "", catalog), "银行")
        self.assertEqual(l1_name("801020", "", catalog), None)
        sibling = normalize_sw_catalog(
            pd.DataFrame({"指数代码": ["801010", "801020"], "指数名称": ["农林牧渔", "采掘"]})
        )
        self.assertEqual(l1_name("801020", "", sibling), "采掘")
        self.assertIsNone(l1_name("801099", "", sibling))

    def test_excess_return_removes_the_event_day_risk_free_rate(self):
        n_days = 160
        dates = pd.bdate_range("2021-01-04", periods=n_days)
        rng = np.random.default_rng(3)
        mkt = rng.normal(0.0, 0.01, size=n_days)
        mkt[0] = 0.0
        rf = np.full(n_days, 0.0001)
        rf[120] = 0.01
        shock = np.zeros(n_days)
        shock[120] = 0.02
        rets = rf + 0.1 * mkt + shock
        from ashare_event.schema import load_events

        events = load_events(
            pd.DataFrame(
                [
                    {
                        "code": "000001",
                        "event_date": dates[120],
                        "event_type": "ai_policy",
                        "industry": "银行",
                        "firm_name": "公司",
                    }
                ]
            )
        )
        prices = pd.DataFrame({"code": "000001", "date": dates, "close": 10 * np.cumprod(1.0 + rets)})
        market = pd.DataFrame({"date": dates, "close": np.full(n_days, 1000.0)})
        factors = pd.DataFrame(
            {
                "date": dates,
                "mkt": mkt,
                "smb": np.zeros(n_days),
                "hml": np.zeros(n_days),
                "rf": rf,
            }
        )
        fitted = _fit(events, prices, market, model="ff3", factors_frame=factors)
        self.assertEqual(fitted.dropped["reason"].tolist(), [])
        self.assertAlmostEqual(float(fitted.events["car_p0_p0"].iloc[0]), 0.02, places=3)
        self.assertAlmostEqual(float(fitted.events["beta"].iloc[0]), 0.1, places=3)

    def test_limit_day_drop_keeps_the_other_event_and_sums(self):
        n_days = 160
        dates = pd.bdate_range("2021-01-04", periods=n_days)
        market = pd.DataFrame({"date": dates, "close": np.linspace(1000, 1010, n_days)})
        rows = []
        frames = []
        for index, shock in enumerate((0.099, 0.01)):
            rets = np.full(n_days, 0.001)
            rets[0] = 0.0
            rets[120] = shock
            code = f"{index + 1:06d}"
            frames.append(pd.DataFrame({"code": code, "date": dates, "close": 10 * np.cumprod(1.0 + rets)}))
            rows.append(
                {
                    "code": code,
                    "event_date": dates[120],
                    "event_type": "ai_policy",
                    "industry": "软件",
                    "firm_name": f"公司{index + 1}",
                }
            )
        from ashare_event.schema import load_events

        events = load_events(pd.DataFrame(rows))
        fitted = _fit(events, pd.concat(frames, ignore_index=True), market, drop_event_day_limit=True)
        self.assertEqual(len(fitted.events), 1)
        self.assertEqual(fitted.events["code"].iloc[0], "000002")
        self.assertEqual(fitted.dropped["reason"].tolist(), ["limit_day"])
        _dropped, attrition = _sample_account(2, pd.DataFrame(), fitted, "zh")
        self.assertEqual(int(attrition.loc[attrition["reason"] != "loaded", "n"].sum()), 2)
        self.assertIn("事件日涨跌停", set(attrition["step"]))

    def test_membership_integer_codes_keep_leading_zeros(self):
        frame = pd.DataFrame(
            {
                "股票代码": [1, 2230],
                "计入日期": ["2021-07-30", "2021-07-30"],
                "行业代码": [480301, 710101],
            }
        )
        out = normalize_membership(frame)
        self.assertEqual(out["code"].tolist(), ["000001", "002230"])
        self.assertEqual(out["industry_code"].tolist(), ["480301", "710101"])
        self.assertEqual(class_l1_index("710101"), "801750")

    def test_case_file_has_no_example_marker(self):
        events = load_case_events()
        self.assertEqual(len(events), 4)
        self.assertFalse(events["firm_name"].astype(str).str.contains("示例").any())
        self.assertEqual(set(pd.to_datetime(events["event_date"]).dt.strftime("%Y-%m-%d")), {"2023-07-13"})


if __name__ == "__main__":
    unittest.main()
