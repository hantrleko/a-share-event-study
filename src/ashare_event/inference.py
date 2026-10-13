"""Tests on mean CAR.

Cross-sectional t assumes independent events. BMP (Boehmer, Musumeci, and
Poulsen 1991) lets event-window variance differ from the estimation window.
Kolari and Pynnonen (2010) shrink that t by the average residual correlation.
When half or more of the events share a calendar date, the portfolio
time-series test is the preferred one: the day is a single time-series shock.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ashare_event.returns import _fit_market_model, car_variance


def _two_sided(stat: float, df: float) -> float:
    if not np.isfinite(stat) or df <= 0:
        return np.nan
    return float(2 * stats.t.sf(abs(stat), df))


def mean_t(values: np.ndarray) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    n_obs = int(values.size)
    if n_obs < 2:
        mean = float(values.mean()) if n_obs else np.nan
        return mean, np.nan, np.nan
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n_obs))
    t_stat = mean / se if se > 0 else np.nan
    return mean, t_stat, _two_sided(t_stat, n_obs - 1)


def kp_factor(r_bar: float, n_obs: int) -> float:
    """Multiplier applied to a t statistic. Values below 1 widen the test."""
    if n_obs < 2 or not np.isfinite(r_bar):
        return 1.0
    lower = -0.99 / (n_obs - 1)
    r_value = float(np.clip(r_bar, lower + 1e-8, 0.99))
    return float(np.sqrt((1.0 - r_value) / (1.0 + (n_obs - 1) * r_value)))


def mean_residual_correlation(panel: pd.DataFrame, event_ids: pd.Series, min_periods: int) -> float:
    if panel.empty or len(event_ids) < 3:
        return 0.0
    resid = panel.loc[panel["event_id"].isin(set(event_ids)) & panel["in_est"], ["date", "event_id", "resid"]]
    resid = resid.dropna(subset=["resid"])
    if resid.empty:
        return 0.0
    wide = resid.pivot(index="date", columns="event_id", values="resid")
    corr = wide.corr(min_periods=min_periods)
    if corr.shape[0] < 2:
        return 0.0
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    values = upper.stack().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def portfolio_t(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    window: tuple[int, int],
    estimation_window: tuple[int, int],
    min_share: float,
    model: str,
) -> tuple[float, float, float]:
    """Return (t, p, modal share). t is NaN when dates are not concentrated."""
    if events.empty:
        return np.nan, np.nan, np.nan
    dates = pd.to_datetime(events["event_date_aligned"])
    modal = dates.mode().iloc[0]
    share = float((dates == modal).mean())
    if share < min_share:
        return np.nan, np.nan, share
    ids = set(events.loc[dates == modal, "event_id"])
    sub = panel.loc[panel["event_id"].isin(ids), ["rel_day", "ret", "mkt_ret"]].dropna()
    if sub.empty:
        return np.nan, np.nan, share
    portfolio = sub.groupby("rel_day", as_index=False)[["ret", "mkt_ret"]].mean()
    est_left, est_right = estimation_window
    est = portfolio[(portfolio["rel_day"] >= est_left) & (portfolio["rel_day"] <= est_right)]
    fit = _fit_market_model(est["ret"].to_numpy(dtype=float), est["mkt_ret"].to_numpy(dtype=float), model)
    if fit is None:
        return np.nan, np.nan, share
    left, right = window
    event = portfolio[(portfolio["rel_day"] >= left) & (portfolio["rel_day"] <= right)]
    ar = event["ret"].to_numpy(dtype=float) - (fit["alpha"] + fit["beta"] * event["mkt_ret"].to_numpy(dtype=float))
    if not np.isfinite(ar).all() or ar.size == 0:
        return np.nan, np.nan, share
    var_car = car_variance(fit, event["mkt_ret"].to_numpy(dtype=float))
    if not np.isfinite(var_car) or var_car <= 0:
        return np.nan, np.nan, share
    t_stat = float(np.sum(ar) / np.sqrt(var_car))
    return t_stat, _two_sided(t_stat, fit["df"]), share


def preferred_name(portfolio: float, r_bar: float) -> str:
    if np.isfinite(portfolio):
        return "portfolio"
    if np.isfinite(r_bar) and r_bar > 0.05:
        return "kp"
    return "bmp"


def summarize_window(
    events: pd.DataFrame,
    panel: pd.DataFrame,
    window: tuple[int, int],
    suffix: str,
    *,
    estimation_window: tuple[int, int],
    model: str,
    r_bar: float,
    min_share: float,
) -> dict:
    car_col = f"car_{suffix}"
    scar_col = f"scar_{suffix}"
    sample = events.dropna(subset=[car_col]).copy()
    mean, t_cs, p_cs = mean_t(sample[car_col].to_numpy(dtype=float))
    _, t_bmp, p_bmp = mean_t(sample[scar_col].to_numpy(dtype=float))
    n_obs = int(sample[car_col].notna().sum())
    factor = kp_factor(r_bar, n_obs)
    t_kp = float(t_bmp * factor) if np.isfinite(t_bmp) else np.nan
    p_kp = _two_sided(t_kp, max(n_obs - 1, 1))
    t_port, p_port, share = portfolio_t(
        panel, sample, window, estimation_window, min_share, model
    )
    choice = preferred_name(t_port, r_bar)
    p_map = {"portfolio": p_port, "kp": p_kp, "bmp": p_bmp}
    return {
        "t1": window[0],
        "t2": window[1],
        "n": n_obs,
        "mean": mean,
        "t_cs": t_cs,
        "p_cs": p_cs,
        "t_bmp": t_bmp,
        "p_bmp": p_bmp,
        "t_kp": t_kp,
        "p_kp": p_kp,
        "t_portfolio": t_port,
        "p_portfolio": p_port,
        "r_bar": r_bar,
        "kp_factor": factor,
        "modal_share": share,
        "preferred": choice,
        "p_preferred": p_map[choice],
    }
