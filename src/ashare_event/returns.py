"""Market-model abnormal returns and cumulative abnormal returns.

Estimand: the firm's simple return minus the return predicted from its own
pre-event market model (MacKinlay 1997). This is not a treatment effect.
The estimation window is required to end before every CAR window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ashare_event.config import window_suffix


@dataclass
class Estimation:
    panel: pd.DataFrame
    events: pd.DataFrame
    dropped: pd.DataFrame
    calendar: pd.DatetimeIndex


def _fit_market_model(y: np.ndarray, x: np.ndarray, model: str) -> dict | None:
    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]
    n_obs = int(y.size)
    if n_obs < 3:
        return None
    if model == "market_adjusted":
        resid = y - x
        sigma2 = float(np.sum(resid**2) / (n_obs - 1))
        return {
            "alpha": 0.0,
            "beta": 1.0,
            "sigma2": sigma2,
            "x_bar": float(np.mean(x)),
            "sxx": float(np.sum((x - np.mean(x)) ** 2)),
            "n_est": n_obs,
            "df": n_obs - 1,
            "kind": "market_adjusted",
        }
    design = np.column_stack([np.ones(n_obs), x])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    sigma2 = float(np.sum(resid**2) / (n_obs - 2))
    x_bar = float(np.mean(x))
    return {
        "alpha": float(coef[0]),
        "beta": float(coef[1]),
        "sigma2": sigma2,
        "x_bar": x_bar,
        "sxx": float(np.sum((x - x_bar) ** 2)),
        "n_est": n_obs,
        "df": n_obs - 2,
        "kind": "market",
    }


def car_variance(fit: dict, market: np.ndarray) -> float:
    """MacKinlay (1997) forecast-error variance of a CAR."""
    length = int(market.size)
    if length == 0 or fit["sigma2"] <= 0:
        return np.nan
    if fit.get("kind") == "market_adjusted":
        return float(fit["sigma2"] * length)
    deviation = float(np.sum(market - fit["x_bar"]))
    sxx = fit["sxx"] if fit["sxx"] > 0 else np.inf
    return float(fit["sigma2"] * (length + (length**2) / fit["n_est"] + (deviation**2) / sxx))


def _returns_and_calendar(
    prices: pd.DataFrame, market: pd.DataFrame
) -> tuple[dict[str, pd.Series], pd.Series, pd.DatetimeIndex]:
    market = market.sort_values("date").drop_duplicates("date").copy()
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    market["mkt_ret"] = market["close"].pct_change()
    mkt = market.dropna(subset=["mkt_ret"]).set_index("date")["mkt_ret"]
    prices = prices.sort_values(["code", "date"]).drop_duplicates(["code", "date"]).copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices["code"] = prices["code"].astype(str).str.zfill(6).str[-6:]
    prices["ret"] = prices.groupby("code", sort=False)["close"].pct_change()
    by_code = {
        code: part.dropna(subset=["ret"]).drop_duplicates("date").set_index("date")["ret"]
        for code, part in prices.groupby("code", sort=False)
    }
    calendar = pd.DatetimeIndex(mkt.index.sort_values().unique())
    mkt = mkt.reindex(calendar)
    return by_code, mkt, calendar


def _align(calendar: pd.DatetimeIndex, event_date: pd.Timestamp) -> tuple[int, pd.Timestamp] | None:
    event_date = pd.Timestamp(event_date).normalize()
    pos = int(calendar.searchsorted(event_date))
    if pos >= len(calendar):
        return None
    aligned = pd.Timestamp(calendar[pos])
    if (aligned - event_date).days > 10:
        return None
    return pos, aligned


def estimate_events(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    market: pd.DataFrame,
    *,
    estimation_window: tuple[int, int],
    event_windows: list[tuple[int, int]],
    path_window: tuple[int, int],
    regression_window: tuple[int, int],
    model: str = "market",
    min_estimation_obs: int = 80,
    require_complete_window: bool = True,
) -> Estimation:
    by_code, mkt, calendar = _returns_and_calendar(prices, market)
    est_left, est_right = estimation_window
    windows = list(dict.fromkeys([*event_windows, tuple(regression_window)]))
    slice_left = min(est_left, path_window[0], min(window[0] for window in windows))
    slice_right = max(path_window[1], max(window[1] for window in windows))
    kept_rows: list[dict] = []
    dropped_rows: list[dict] = []
    panel_rows: list[pd.DataFrame] = []

    for event in events.itertuples(index=False):
        code = str(event.code).zfill(6)[-6:]
        firm_ret = by_code.get(code)
        if firm_ret is None or firm_ret.empty:
            dropped_rows.append({"event_id": event.event_id, "reason": "no_price"})
            continue
        aligned = _align(calendar, event.event_date)
        if aligned is None:
            dropped_rows.append({"event_id": event.event_id, "reason": "outside_calendar"})
            continue
        pos, aligned_date = aligned
        left = pos + slice_left
        right = pos + slice_right
        if pos + est_left < 0 or pos + est_right >= len(calendar):
            dropped_rows.append({"event_id": event.event_id, "reason": "short_history"})
            continue
        if left < 0:
            left = 0
        right = min(right, len(calendar) - 1)
        dates = calendar[left : right + 1]
        rel = np.arange(left, right + 1) - pos
        window_frame = pd.DataFrame(
            {
                "date": dates,
                "rel_day": rel,
                "ret": firm_ret.reindex(dates).to_numpy(dtype=float),
                "mkt_ret": mkt.reindex(dates).to_numpy(dtype=float),
            }
        )
        est_mask = (window_frame["rel_day"] >= est_left) & (window_frame["rel_day"] <= est_right)
        fit = _fit_market_model(
            window_frame.loc[est_mask, "ret"].to_numpy(dtype=float),
            window_frame.loc[est_mask, "mkt_ret"].to_numpy(dtype=float),
            model,
        )
        if fit is None or fit["n_est"] < min_estimation_obs:
            dropped_rows.append({"event_id": event.event_id, "reason": "few_estimation_obs"})
            continue
        predicted = fit["alpha"] + fit["beta"] * window_frame["mkt_ret"]
        window_frame["ar"] = window_frame["ret"] - predicted
        window_frame["resid"] = np.where(est_mask, window_frame["ar"], np.nan)
        window_frame["event_id"] = event.event_id
        window_frame["firm_id"] = event.firm_id
        window_frame["in_est"] = est_mask.to_numpy()

        record = {
            "event_id": event.event_id,
            "alpha": fit["alpha"],
            "beta": fit["beta"],
            "sigma2": fit["sigma2"],
            "n_est": fit["n_est"],
            "event_date_aligned": aligned_date,
            "rolled_forward": bool(aligned_date != pd.Timestamp(event.event_date).normalize()),
        }
        primary_ok = False
        for left_w, right_w in windows:
            piece = window_frame[(window_frame["rel_day"] >= left_w) & (window_frame["rel_day"] <= right_w)]
            complete = piece["ar"].notna().all() and len(piece) == (right_w - left_w + 1)
            key = window_suffix(left_w, right_w)
            if require_complete_window and not complete:
                record[f"car_{key}"] = np.nan
                record[f"var_{key}"] = np.nan
                record[f"scar_{key}"] = np.nan
            else:
                ar = piece["ar"].to_numpy(dtype=float)
                rm = piece["mkt_ret"].to_numpy(dtype=float)
                usable = np.isfinite(ar) & np.isfinite(rm)
                car = float(np.sum(ar[usable])) if usable.any() else np.nan
                var_car = car_variance(fit, rm[usable]) if usable.any() else np.nan
                scar = car / np.sqrt(var_car) if np.isfinite(var_car) and var_car > 0 else np.nan
                record[f"car_{key}"] = car
                record[f"var_{key}"] = var_car
                record[f"scar_{key}"] = scar
            if (left_w, right_w) == tuple(regression_window) and np.isfinite(record[f"car_{key}"]):
                primary_ok = True
        path = window_frame[(window_frame["rel_day"] >= path_window[0]) & (window_frame["rel_day"] <= path_window[1])]
        path_complete = path["ar"].notna().all() and len(path) == (path_window[1] - path_window[0] + 1)
        record["path_complete"] = bool(path_complete)
        if not primary_ok:
            dropped_rows.append({"event_id": event.event_id, "reason": "incomplete_window"})
            continue
        window_frame["car_path"] = np.nan
        if path_complete:
            ordered = path.sort_values("rel_day")
            window_frame.loc[ordered.index, "car_path"] = ordered["ar"].cumsum().to_numpy()
        panel_rows.append(window_frame)
        kept_rows.append(record)

    generated = {"alpha", "beta", "sigma2", "n_est", "event_date_aligned", "rolled_forward", "path_complete"}
    meta_cols = [
        col
        for col in events.columns
        if col not in generated and not col.startswith(("car_", "var_", "scar_"))
    ]
    base = events[meta_cols].copy()
    if kept_rows:
        kept = base.merge(pd.DataFrame(kept_rows), on="event_id", how="inner")
    else:
        kept = base.iloc[0:0].copy()
    dropped = pd.DataFrame(dropped_rows, columns=["event_id", "reason"])
    long = pd.concat(panel_rows, ignore_index=True) if panel_rows else pd.DataFrame()
    return Estimation(panel=long, events=kept, dropped=dropped, calendar=calendar)


def add_peer_count(events: pd.DataFrame, days: int = 5) -> pd.DataFrame:
    """Count other analysis-sample events in the same industry within ±days.

    The count is endogenous and is bounded by the analysis sample, not the market.
    """
    out = events.copy()
    if out.empty:
        out["peer_count_pm5"] = pd.Series(dtype=int)
        return out
    dates = pd.to_datetime(out["event_date_aligned"])
    counts = []
    for idx, row in out.iterrows():
        gap = (dates - dates.loc[idx]).abs()
        mask = (out["industry"] == row["industry"]) & (out["event_id"] != row["event_id"]) & (gap <= pd.Timedelta(days=days))
        counts.append(int(mask.sum()))
    out["peer_count_pm5"] = counts
    return out
