"""Simulated A-share calendar, prices, and events.

The DGP is a market model plus a one-day shock on the event date.
Output is labeled simulated wherever it is exported. It is not evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_event.config import StudyConfig
from ashare_event.schema import load_events


def simulate(cfg: StudyConfig, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    est_left = cfg.estimation_window[0]
    post = max(cfg.path_window[1], max(window[1] for window in cfg.event_windows))
    pre = abs(est_left) + abs(min(cfg.placebo_offsets[0], 0)) + 40
    event_loc = pre
    n_days = event_loc + post + 40
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    market_ret = rng.normal(0.0003, 0.011, size=n_days)
    market_ret[0] = 0.0
    market = pd.DataFrame(
        {"date": dates, "close": 4000 * np.cumprod(1.0 + market_ret)}
    )

    industries = ["银行", "电子", "医药", "公用事业"]
    event_types = ["ai_disclosure", "esg", "geopolitics"]
    n_firms = 48
    price_frames = []
    rows = []
    for i in range(n_firms):
        event_type = event_types[i % 3]
        signal_cost = float(rng.uniform(0.05, 0.95))
        legitimacy_gap = float(rng.uniform(0.05, 0.95))
        size_log = float(rng.normal(15.5, 0.8))
        beta = float(rng.uniform(0.8, 1.2))
        if event_type == "geopolitics":
            loc = event_loc
            shock = -0.02
        elif event_type == "ai_disclosure":
            loc = event_loc - (i % 15)
            shock = 0.008 + 0.025 * signal_cost
        else:
            loc = event_loc - (i % 12)
            shock = -0.01 - 0.02 * legitimacy_gap
        eps = rng.normal(0.0, 0.012, size=n_days)
        rets = 0.0 + beta * market_ret + eps
        rets[0] = 0.0
        rets[loc] += shock
        code = f"{i + 1:06d}"
        price_frames.append(
            pd.DataFrame({"code": code, "date": dates, "close": 20 * np.cumprod(1.0 + rets)})
        )
        rows.append(
            {
                "firm_id": f"DEMO_{code}",
                "code": code,
                "event_date": dates[loc],
                "event_type": event_type,
                "industry": industries[i % 4],
                "firm_name": f"演示公司{i + 1}",
                "signal_cost": signal_cost,
                "size_log": size_log,
                "legitimacy_gap": legitimacy_gap,
                "beta_true": beta,
                "shock_true": shock,
            }
        )
    events = load_events(pd.DataFrame(rows))
    prices = pd.concat(price_frames, ignore_index=True)
    return events, prices, market
