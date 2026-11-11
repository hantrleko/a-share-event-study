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

FACTOR_COLUMNS = {
    "ff3": ("mkt", "smb", "hml"),
    "ff5": ("mkt", "smb", "hml", "rmw", "cma"),
    "carhart": ("mkt", "smb", "hml", "umd"),
}
_MODELS = {"market", "market_adjusted", "industry", *FACTOR_COLUMNS}


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


def _fit_dimson(y: np.ndarray, x: np.ndarray) -> dict | None:
    """Market model with one lead and one lag. Beta is the sum of the three slopes.

    The event-window forecast is intercept plus that sum times the same-day
    market return. Leads are not used after the estimation window.
    Variance stays the single-index formula, with this regression's sigma2
    and the contemporaneous market sum of squares.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 5 or x.size != y.size:
        return None
    response = y[1:-1]
    design_x = np.column_stack([x[:-2], x[1:-1], x[2:]])
    mask = np.isfinite(response) & np.isfinite(design_x).all(axis=1)
    response = response[mask]
    design_x = design_x[mask]
    n_obs = int(response.size)
    if n_obs < 8:
        return None
    design = np.column_stack([np.ones(n_obs), design_x])
    coef, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    resid = response - design @ coef
    df = n_obs - 4
    if df <= 0:
        return None
    current = design_x[:, 1]
    x_bar = float(np.mean(current))
    return {
        "alpha": float(coef[0]),
        "beta": float(coef[1] + coef[2] + coef[3]),
        "sigma2": float(np.sum(resid**2) / df),
        "x_bar": x_bar,
        "sxx": float(np.sum((current - x_bar) ** 2)),
        "n_est": n_obs,
        "df": df,
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
    market_lags: int = 0,
) -> float:
    """CAR at one calendar index. The estimation window moves with ``pos``."""
    n_obs = int(firm_ret.shape[0])
    est_left, est_right = estimation_window
    if pos < 0 or pos + est_left < 0 or pos + est_right >= n_obs:
        return np.nan
    y = firm_ret[pos + est_left : pos + est_right + 1]
    if model in FACTOR_COLUMNS:
        if factors is None:
            return np.nan
        fit = fit_factor_model(y, factors[pos + est_left : pos + est_right + 1])
    else:
        kind = "market_adjusted" if model == "market_adjusted" else "market"
        benchmark_est = benchmark[pos + est_left : pos + est_right + 1]
        if market_lags and kind == "market":
            fit = _fit_dimson(y, benchmark_est)
        else:
            fit = _fit_market_model(y, benchmark_est, kind)
    if fit is None or fit["n_est"] < min_estimation_obs:
        return np.nan
    left, right = event_window
    if pos + left < 0 or pos + right >= n_obs:
        return np.nan
    y_event = firm_ret[pos + left : pos + right + 1]
    if model in FACTOR_COLUMNS:
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
    market_lags: int = 0,
    drop_event_day_limit: bool = False,
) -> Estimation:
    if model not in _MODELS:
        raise ValueError("model 只能是 market、market_adjusted、industry、ff3、ff5 或 carhart。")
    by_code, mkt, calendar = _returns_and_calendar(prices, market)
    industry_map: dict[str, pd.Series] = {}
    factor_indexed = None
    factor_names: tuple[str, ...] = ()
    if model == "industry":
        from ashare_event.market import prepare_industry_returns

        industry_map = prepare_industry_returns(industry_frame)
        if not industry_map:
            raise ValueError("行业基准需要行业指数行情，列至少包括行业、日期、收盘。")
    if model in FACTOR_COLUMNS:
        from ashare_event.market import prepare_factors

        factor_names = FACTOR_COLUMNS[model]
        factors = prepare_factors(factors_frame)
        missing = [] if factors is None else [name for name in factor_names if name not in factors.columns]
        if factors is None or missing:
            if model == "ff5":
                raise ValueError("五因子模型需要日期、mkt、smb、hml、rmw、cma。内置 CH3 没有 RMW 和 CMA。")
            if model == "carhart":
                raise ValueError("Carhart 模型需要日期、mkt、smb、hml、umd。内置 CH3 没有 UMD。")
            raise ValueError("三因子模型需要日期、mkt、smb、hml。")
        keep_factor_cols = [*factor_names, *(["rf"] if "rf" in factors.columns else [])]
        factor_indexed = factors.set_index("date")[keep_factor_cols]
    est_left, est_right = estimation_window
    windows = list(dict.fromkeys([*event_windows, tuple(regression_window)]))
    slice_left = min(est_left, path_window[0], min(window[0] for window in windows))
    slice_right = max(path_window[1], max(window[1] for window in windows))
    kept_rows: list[dict] = []
    dropped_rows: list[dict] = []
    panel_rows: list[pd.DataFrame] = []

    def _drop(event, reason: str, failed_window: str = "") -> None:
        name = getattr(event, "firm_name", "")
        if name is None or (isinstance(name, float) and np.isnan(name)):
            name = ""
        stamp = pd.to_datetime(getattr(event, "event_date", None), errors="coerce")
        dropped_rows.append(
            {
                "event_id": event.event_id,
                "code": str(event.code).zfill(6)[-6:],
                "event_date": "" if pd.isna(stamp) else pd.Timestamp(stamp).strftime("%Y-%m-%d"),
                "firm_name": str(name),
                "reason": reason,
                "failed_window": failed_window,
            }
        )

    for event in events.itertuples(index=False):
        code = str(event.code).zfill(6)[-6:]
        firm_ret = by_code.get(code)
        if firm_ret is None or firm_ret.empty:
            _drop(event, "no_price")
            continue
        aligned = _align(calendar, event.event_date, after=parse_after_close(getattr(event, "after_close", False)))
        if aligned is None:
            _drop(event, "outside_calendar")
            continue
        pos, aligned_date = aligned
        left = pos + slice_left
        right = pos + slice_right
        if pos + est_left < 0 or pos + est_right >= len(calendar):
            _drop(event, "short_history")
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
            sw_name = str(getattr(event, "sw_l1", "") or "")
            if sw_name in {"", "nan", "None"}:
                sw_name = str(getattr(event, "industry", ""))
            series = industry_map.get(sw_name)
            if series is None:
                _drop(event, "no_industry_benchmark")
                continue
            window_frame["mkt_ret"] = series.reindex(window_frame["date"]).to_numpy(dtype=float)
        fac_cols = [f"fac_{name}" for name in factor_names]
        has_rf = bool(factor_names) and factor_indexed is not None and "rf" in factor_indexed.columns
        if factor_names:
            fac = factor_indexed.reindex(window_frame["date"])
            for name in factor_names:
                window_frame[f"fac_{name}"] = fac[name].to_numpy(dtype=float)
            window_frame["mkt_ret"] = window_frame["fac_mkt"]
            if has_rf:
                window_frame["rf"] = fac["rf"].to_numpy(dtype=float)
        est_mask = (window_frame["rel_day"] >= est_left) & (window_frame["rel_day"] <= est_right)
        if factor_names:
            y_est = window_frame.loc[est_mask, "ret"].to_numpy(dtype=float)
            if has_rf:
                y_est = y_est - window_frame.loc[est_mask, "rf"].to_numpy(dtype=float)
            fit = fit_factor_model(y_est, window_frame.loc[est_mask, fac_cols].to_numpy(dtype=float))
        else:
            fit_kind = "market_adjusted" if model == "market_adjusted" else "market"
            y_est = window_frame.loc[est_mask, "ret"].to_numpy(dtype=float)
            x_est = window_frame.loc[est_mask, "mkt_ret"].to_numpy(dtype=float)
            if market_lags and fit_kind == "market":
                fit = _fit_dimson(y_est, x_est)
            else:
                fit = _fit_market_model(y_est, x_est, fit_kind)
        if fit is None or fit["n_est"] < min_estimation_obs:
            _drop(event, "few_estimation_obs")
            continue
        if factor_names:
            factor_x = window_frame[fac_cols].to_numpy(dtype=float)
            predicted = fit["alpha"] + factor_x @ fit["betas"]
        else:
            predicted = fit["alpha"] + fit["beta"] * window_frame["mkt_ret"]
        window_frame["predicted"] = predicted
        observed = window_frame["ret"].to_numpy(dtype=float)
        if has_rf:
            observed = observed - window_frame["rf"].to_numpy(dtype=float)
        window_frame["ar"] = observed - window_frame["predicted"]
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
        band_hit = bool(is_limit_hit(ret0, band))
        record["limit_hit"] = band_hit
        exchange_hit = bool(getattr(event, "limit_exchange", False))
        if drop_event_day_limit and (exchange_hit or band_hit):
            _drop(event, "limit_day")
            continue
        span_source = [item for item in windows if int(item[1]) >= 0] or list(windows)
        span_left = min(int(item[0]) for item in span_source)
        span_right = max(int(item[1]) for item in span_source)
        span = window_frame[(window_frame["rel_day"] >= span_left) & (window_frame["rel_day"] <= span_right)]
        span_rets = span["ret"].to_numpy(dtype=float)
        span_hits = [is_limit_hit(float(value), band) for value in span_rets if np.isfinite(value)]
        record["limit_hit_days"] = int(sum(span_hits))
        record["limit_capped_end"] = bool(span_hits[-1]) if span_hits else False
        if fit.get("kind") == "factor":
            for index, name in enumerate(factor_names):
                if index == 0:
                    continue
                record[f"beta_{name}"] = float(fit["betas"][index])
        primary_ok = False
        for left_w, right_w in windows:
            piece = window_frame[(window_frame["rel_day"] >= left_w) & (window_frame["rel_day"] <= right_w)]
            complete = piece["ar"].notna().all() and len(piece) == (right_w - left_w + 1)
            key = window_suffix(left_w, right_w)
            raw_ret = piece["ret"].to_numpy(dtype=float)
            if has_rf:
                raw_ret = raw_ret - piece["rf"].to_numpy(dtype=float)
            raw_pred = piece["predicted"].to_numpy(dtype=float)
            buy_hold = np.isfinite(raw_ret) & np.isfinite(raw_pred)
            if require_complete_window and not complete:
                record[f"car_{key}"] = np.nan
                record[f"var_{key}"] = np.nan
                record[f"scar_{key}"] = np.nan
                record[f"bhar_{key}"] = np.nan
            else:
                ar = piece["ar"].to_numpy(dtype=float)
                rm = piece["mkt_ret"].to_numpy(dtype=float)
                if fit.get("kind") == "factor":
                    factor_piece = piece[fac_cols].to_numpy(dtype=float)
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
                if buy_hold.any():
                    record[f"bhar_{key}"] = float(np.prod(1.0 + raw_ret[buy_hold]) - np.prod(1.0 + raw_pred[buy_hold]))
                else:
                    record[f"bhar_{key}"] = np.nan
            if (left_w, right_w) == tuple(regression_window) and np.isfinite(record[f"car_{key}"]):
                primary_ok = True
        path = window_frame[(window_frame["rel_day"] >= path_window[0]) & (window_frame["rel_day"] <= path_window[1])]
        path_complete = path["ar"].notna().all() and len(path) == (path_window[1] - path_window[0] + 1)
        record["path_complete"] = bool(path_complete)
        if not primary_ok:
            _drop(event, "incomplete_window", f"[{regression_window[0]}, {regression_window[1]}]")
            continue
        window_frame["car_path"] = np.nan
        if path_complete:
            ordered = path.sort_values("rel_day")
            window_frame.loc[ordered.index, "car_path"] = ordered["ar"].cumsum().to_numpy()
        panel_rows.append(window_frame.drop(columns=["predicted"]))
        kept_rows.append(record)

    generated = {
        "alpha",
        "beta",
        "sigma2",
        "n_est",
        "event_date_aligned",
        "rolled_forward",
        "path_complete",
        "limit_band",
        "limit_hit",
        "limit_hit_days",
        "limit_capped_end",
    }
    meta_cols = [
        col
        for col in events.columns
        if col not in generated and not col.startswith(("car_", "var_", "scar_", "bhar_", "beta_"))
    ]
    base = events[meta_cols].copy()
    if kept_rows:
        kept = base.merge(pd.DataFrame(kept_rows), on="event_id", how="inner")
    else:
        kept = base.iloc[0:0].copy()
    drop_columns = ["event_id", "code", "event_date", "firm_name", "reason", "failed_window"]
    dropped = pd.DataFrame(dropped_rows, columns=drop_columns) if dropped_rows else pd.DataFrame(columns=drop_columns)
    long = pd.concat(panel_rows, ignore_index=True) if panel_rows else pd.DataFrame()
    return Estimation(panel=long, events=kept, dropped=dropped, calendar=calendar)


def winsorize_events(events: pd.DataFrame, rate: float) -> pd.DataFrame:
    """Clip CAR and BHAR. Rate 0 leaves the frame unchanged. The path is not clipped."""
    rate = float(rate or 0)
    if rate <= 0 or events.empty:
        return events
    out = events.copy()
    for column in list(out.columns):
        if not str(column).startswith(("car_", "bhar_")):
            continue
        values = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size < 8:
            continue
        low, high = np.quantile(finite, [rate, 1.0 - rate])
        clipped = values.copy()
        mask = np.isfinite(clipped)
        clipped[mask] = np.clip(clipped[mask], low, high)
        out[column] = clipped
        if not str(column).startswith("car_"):
            continue
        key = str(column)[4:]
        var_col = f"var_{key}"
        scar_col = f"scar_{key}"
        if var_col not in out.columns or scar_col not in out.columns:
            continue
        variance = pd.to_numeric(out[var_col], errors="coerce").to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            scar = np.where(
                np.isfinite(variance) & (variance > 0) & np.isfinite(clipped),
                clipped / np.sqrt(variance),
                np.nan,
            )
        out[scar_col] = scar
    return out


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
