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
from ashare_event.schema import parse_after_close


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


def limit_band(code: str, firm_name: str = "") -> float:
    """Heuristic daily price-limit band. Not an exchange status file.

    Main board 10%, ChiNext and STAR 20%, Beijing 30%, and a name containing
    ST uses 5%. Listing-day rules and intraday ST changes are not observed.
    """
    name = str(firm_name).upper().replace(" ", "")
    if "ST" in name:
        return 0.05
    text = str(code).zfill(6)[-6:]
    if text.startswith(("300", "301", "688", "689")):
        return 0.20
    if text.startswith("920") or text.startswith(("8", "4")):
        return 0.30
    return 0.10


def is_limit_hit(ret: float, band: float) -> bool:
    """True when a simple return sits within 0.5 percentage point of the band."""
    if not np.isfinite(ret) or not np.isfinite(band):
        return False
    return abs(float(ret)) + 1e-12 >= float(band) - 0.005


def fit_factor_model(y: np.ndarray, x: np.ndarray) -> dict | None:
    """Intercept plus the columns of x. x does not include the intercept."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    y = y[mask]
    x = x[mask]
    n_obs, k = (int(y.size), int(x.shape[1])) if y.size else (0, int(x.shape[1]) if x.ndim == 2 else 0)
    if n_obs < k + 3:
        return None
    design = np.column_stack([np.ones(n_obs), x])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    df = n_obs - int(design.shape[1])
    if df <= 0:
        return None
    sigma2 = float(np.sum(resid**2) / df)
    return {
        "alpha": float(coef[0]),
        "beta": float(coef[1]),
        "betas": np.asarray(coef[1:], dtype=float),
        "sigma2": sigma2,
        "x_bar": float(np.mean(x[:, 0])),
        "sxx": float(np.sum((x[:, 0] - np.mean(x[:, 0])) ** 2)),
        "n_est": n_obs,
        "df": df,
        "kind": "factor",
        "xtx_inv": np.linalg.pinv(design.T @ design),
    }


def factor_car_variance(fit: dict, x_event: np.ndarray) -> float:
    """Forecast-error variance of a summed factor-model abnormal return."""
    x_event = np.asarray(x_event, dtype=float)
    if x_event.ndim == 1:
        x_event = x_event.reshape(-1, 1)
    mask = np.isfinite(x_event).all(axis=1)
    x_event = x_event[mask]
    length = int(x_event.shape[0])
    if length == 0 or not np.isfinite(fit.get("sigma2", np.nan)) or fit["sigma2"] <= 0:
        return np.nan
    design = np.column_stack([np.ones(length), x_event])
    iota = np.ones(length)
    bread = np.eye(length) + design @ fit["xtx_inv"] @ design.T
    return float(fit["sigma2"] * iota @ bread @ iota)


def car_on_position(
    firm_ret: np.ndarray,
    benchmark: np.ndarray,
    pos: int,
    estimation_window: tuple[int, int],
    event_window: tuple[int, int],
    model: str,
    min_estimation_obs: int,
    require_complete_window: bool,
    factors: np.ndarray | None = None,
) -> float:
    """CAR at one calendar index. The estimation window moves with ``pos``."""
    n_obs = int(firm_ret.shape[0])
    est_left, est_right = estimation_window
    if pos < 0 or pos + est_left < 0 or pos + est_right >= n_obs:
        return np.nan
    y = firm_ret[pos + est_left : pos + est_right + 1]
    if model == "ff3":
        if factors is None:
            return np.nan
        fit = fit_factor_model(y, factors[pos + est_left : pos + est_right + 1])
    else:
        kind = "market_adjusted" if model == "market_adjusted" else "market"
        fit = _fit_market_model(y, benchmark[pos + est_left : pos + est_right + 1], kind)
    if fit is None or fit["n_est"] < min_estimation_obs:
        return np.nan
    left, right = event_window
    if pos + left < 0 or pos + right >= n_obs:
        return np.nan
    y_event = firm_ret[pos + left : pos + right + 1]
    if model == "ff3":
        x_event = factors[pos + left : pos + right + 1]
        predicted = fit["alpha"] + x_event @ fit["betas"]
        finite = np.isfinite(y_event) & np.isfinite(x_event).all(axis=1)
    else:
        x_event = benchmark[pos + left : pos + right + 1]
        predicted = fit["alpha"] + fit["beta"] * x_event
        finite = np.isfinite(y_event) & np.isfinite(x_event)
    expected = right - left + 1
    if require_complete_window and (y_event.size != expected or not bool(finite.all())):
        return np.nan
    if not bool(finite.any()):
        return np.nan
    return float(np.sum(y_event[finite] - predicted[finite]))


def _align(
    calendar: pd.DatetimeIndex,
    event_date: pd.Timestamp,
    after: bool = False,
) -> tuple[int, pd.Timestamp] | None:
    origin = pd.Timestamp(event_date).normalize()
    probe = origin + pd.Timedelta(days=1) if after else origin
    pos = int(calendar.searchsorted(probe))
    if pos >= len(calendar):
        return None
    aligned = pd.Timestamp(calendar[pos])
    if (aligned - origin).days > 10:
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
    industry_frame: pd.DataFrame | None = None,
    factors_frame: pd.DataFrame | None = None,
) -> Estimation:
    if model not in {"market", "market_adjusted", "industry", "ff3"}:
        raise ValueError("model 只能是 market、market_adjusted、industry 或 ff3。")
    by_code, mkt, calendar = _returns_and_calendar(prices, market)
    industry_map: dict[str, pd.Series] = {}
    factor_indexed = None
    if model == "industry":
        from ashare_event.market import prepare_industry_returns

        industry_map = prepare_industry_returns(industry_frame)
        if not industry_map:
            raise ValueError("行业基准需要行业指数行情，列至少包括行业、日期、收盘。程序不会自动匹配申万指数。")
    if model == "ff3":
        from ashare_event.market import prepare_factors

        factors = prepare_factors(factors_frame)
        if factors is None:
            raise ValueError("三因子模型需要日期、mkt、smb、hml。程序不生成因子收益。")
        factor_indexed = factors.set_index("date")[["mkt", "smb", "hml"]]
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
        aligned = _align(calendar, event.event_date, after=parse_after_close(getattr(event, "after_close", False)))
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
        if model == "industry":
            series = industry_map.get(str(getattr(event, "industry", "")))
            if series is None:
                dropped_rows.append({"event_id": event.event_id, "reason": "no_industry_benchmark"})
                continue
            window_frame["mkt_ret"] = series.reindex(window_frame["date"]).to_numpy(dtype=float)
        if model == "ff3":
            fac = factor_indexed.reindex(window_frame["date"])
            window_frame["fac_mkt"] = fac["mkt"].to_numpy(dtype=float)
            window_frame["fac_smb"] = fac["smb"].to_numpy(dtype=float)
            window_frame["fac_hml"] = fac["hml"].to_numpy(dtype=float)
            window_frame["mkt_ret"] = window_frame["fac_mkt"]
        est_mask = (window_frame["rel_day"] >= est_left) & (window_frame["rel_day"] <= est_right)
        if model == "ff3":
            fit = fit_factor_model(
                window_frame.loc[est_mask, "ret"].to_numpy(dtype=float),
                window_frame.loc[est_mask, ["fac_mkt", "fac_smb", "fac_hml"]].to_numpy(dtype=float),
            )
        else:
            fit_kind = "market_adjusted" if model == "market_adjusted" else "market"
            fit = _fit_market_model(
                window_frame.loc[est_mask, "ret"].to_numpy(dtype=float),
                window_frame.loc[est_mask, "mkt_ret"].to_numpy(dtype=float),
                fit_kind,
            )
        if fit is None or fit["n_est"] < min_estimation_obs:
            dropped_rows.append({"event_id": event.event_id, "reason": "few_estimation_obs"})
            continue
        if model == "ff3":
            factor_x = window_frame[["fac_mkt", "fac_smb", "fac_hml"]].to_numpy(dtype=float)
            predicted = fit["alpha"] + factor_x @ fit["betas"]
        else:
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
        band = limit_band(code, str(getattr(event, "firm_name", "")))
        day0 = window_frame.loc[window_frame["rel_day"] == 0, "ret"]
        ret0 = float(day0.iloc[0]) if len(day0) else np.nan
        record["limit_band"] = band
        record["limit_hit"] = bool(is_limit_hit(ret0, band))
        if fit.get("kind") == "factor":
            record["beta_smb"] = float(fit["betas"][1])
            record["beta_hml"] = float(fit["betas"][2])
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
                if fit.get("kind") == "factor":
                    factor_piece = piece[["fac_mkt", "fac_smb", "fac_hml"]].to_numpy(dtype=float)
                    usable = np.isfinite(ar) & np.isfinite(factor_piece).all(axis=1)
                    var_car = factor_car_variance(fit, factor_piece[usable]) if usable.any() else np.nan
                else:
                    usable = np.isfinite(ar) & np.isfinite(rm)
                    var_car = car_variance(fit, rm[usable]) if usable.any() else np.nan
                car = float(np.sum(ar[usable])) if usable.any() else np.nan
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
    The window is calendar days, matching the previous row-by-row definition.
    """
    out = events.copy()
    if out.empty:
        out["peer_count_pm5"] = pd.Series(dtype=int)
        return out
    counts = pd.Series(0, index=out.index, dtype=int)
    delta = np.timedelta64(int(days), "D")
    for _, part in out.groupby("industry", sort=False):
        dated = pd.to_datetime(part["event_date_aligned"]).to_numpy()
        order = np.argsort(dated)
        ordered = dated[order]
        left = np.searchsorted(ordered, ordered - delta, side="left")
        right = np.searchsorted(ordered, ordered + delta, side="right")
        counts.loc[part.index.to_numpy()[order]] = right - left - 1
    out["peer_count_pm5"] = counts.to_numpy()
    return out
