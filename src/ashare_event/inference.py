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

from ashare_event.returns import _fit_market_model, car_variance, factor_car_variance, fit_factor_model

RBAR_MAX_EVENTS = 80


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


def mean_residual_correlation(
    panel: pd.DataFrame,
    event_ids: pd.Series,
    min_periods: int,
    max_events: int = RBAR_MAX_EVENTS,
) -> float:
    if panel.empty or len(event_ids) < 3:
        return 0.0
    ids = pd.Index(pd.unique(event_ids))
    if len(ids) > max_events:
        picked = np.random.default_rng(0).choice(ids.to_numpy(), size=max_events, replace=False)
        event_ids = pd.Series(picked)
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
    if model == "ff3":
        return _portfolio_factor_t(panel, ids, window, estimation_window, share)
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


def _portfolio_factor_t(panel, ids, window, estimation_window, share) -> tuple[float, float, float]:
    columns = ["rel_day", "ret", "fac_mkt", "fac_smb", "fac_hml"]
    if any(column not in panel.columns for column in columns):
        return np.nan, np.nan, share
    sub = panel.loc[panel["event_id"].isin(ids), columns].dropna()
    if sub.empty:
        return np.nan, np.nan, share
    portfolio = sub.groupby("rel_day", as_index=False)[["ret", "fac_mkt", "fac_smb", "fac_hml"]].mean()
    est_left, est_right = estimation_window
    est = portfolio[(portfolio["rel_day"] >= est_left) & (portfolio["rel_day"] <= est_right)]
    fit = fit_factor_model(est["ret"].to_numpy(dtype=float), est[["fac_mkt", "fac_smb", "fac_hml"]].to_numpy(dtype=float))
    if fit is None:
        return np.nan, np.nan, share
    left, right = window
    event = portfolio[(portfolio["rel_day"] >= left) & (portfolio["rel_day"] <= right)]
    x_event = event[["fac_mkt", "fac_smb", "fac_hml"]].to_numpy(dtype=float)
    ar = event["ret"].to_numpy(dtype=float) - (fit["alpha"] + x_event @ fit["betas"])
    if not np.isfinite(ar).all() or ar.size == 0:
        return np.nan, np.nan, share
    var_car = factor_car_variance(fit, x_event)
    if not np.isfinite(var_car) or var_car <= 0:
        return np.nan, np.nan, share
    t_stat = float(np.sum(ar) / np.sqrt(var_car))
    return t_stat, _two_sided(t_stat, fit["df"]), share


def corrado_t(panel: pd.DataFrame, events: pd.DataFrame, window: tuple[int, int]) -> tuple[float, float]:
    """Cross-sectional t of event-window rank deviations.

    Estimation residuals and event-window abnormal returns are ranked together
    inside each event. This is a thick-tail check, not a treatment-effect test.
    """
    if panel.empty or events.empty:
        return np.nan, np.nan
    left, right = window
    columns = ["event_id", "rel_day", "in_est", "resid", "ar"]
    if any(column not in panel.columns for column in columns):
        return np.nan, np.nan
    sub = panel.loc[panel["event_id"].isin(set(events["event_id"])), columns]
    stats_i = []
    for _, part in sub.groupby("event_id", sort=False):
        est = part.loc[part["in_est"], "resid"].to_numpy(dtype=float)
        est = est[np.isfinite(est)]
        event = part.loc[(part["rel_day"] >= left) & (part["rel_day"] <= right), "ar"].to_numpy(dtype=float)
        event = event[np.isfinite(event)]
        if est.size < 10 or event.size == 0:
            continue
        pooled = np.concatenate([est, event])
        ranks = pd.Series(pooled).rank(method="average").to_numpy(dtype=float)
        k = float(pooled.size)
        expected = (k + 1.0) / 2.0
        var = k * (k + 1.0) / 12.0
        if var <= 0:
            continue
        stats_i.append(float((ranks[-event.size :] - expected).sum() / np.sqrt(event.size * var)))
    if len(stats_i) < 2:
        return np.nan, np.nan
    _mean, t_stat, p_value = mean_t(np.asarray(stats_i, dtype=float))
    return t_stat, p_value


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values. NaNs stay NaN and are not counted."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if int(valid.sum()) == 0:
        return out
    pv = p[valid]
    n_obs = int(pv.size)
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = np.empty(n_obs, dtype=float)
    running = 1.0
    for i in range(n_obs - 1, -1, -1):
        running = min(running, float(ranked[i]) * n_obs / (i + 1))
        adjusted[i] = running
    placed = np.empty(n_obs, dtype=float)
    placed[order] = np.clip(adjusted, 0.0, 1.0)
    out[np.flatnonzero(valid)] = placed
    return out


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
    t_rank, p_rank = corrado_t(panel, sample, window)
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
        "t_corrado": t_rank,
        "p_corrado": p_rank,
    }
