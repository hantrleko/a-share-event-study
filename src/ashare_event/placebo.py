"""Placebos for the mean CAR.

In-time moves each firm to another date of its own history and refits the
benchmark on the estimation window that belongs to that pseudo-event. It does
not slide the original abnormal-return series.

In-space reassigns price series across the sample and refits. The null says
this firm's return is not special on this date.

Neither placebo is a control group, and neither delivers an ATT.
The refit uses pre-aligned return arrays so repeated draws do not rebuild
DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ashare_event.config import StudyConfig
from ashare_event.returns import _align, _returns_and_calendar, car_on_position
from ashare_event.schema import parse_after_close


@dataclass
class _Store:
    calendar: pd.DatetimeIndex
    mkt: np.ndarray
    firm: dict[str, np.ndarray]
    industry: dict[str, np.ndarray]
    factors: np.ndarray | None


def _store(prices, market, industry_frame, factors_frame) -> _Store:
    by_code, mkt, calendar = _returns_and_calendar(prices, market)
    firm = {code: series.reindex(calendar).to_numpy(dtype=float) for code, series in by_code.items()}
    industry: dict[str, np.ndarray] = {}
    if industry_frame is not None:
        from ashare_event.market import prepare_industry_returns

        for name, series in prepare_industry_returns(industry_frame).items():
            industry[name] = series.reindex(calendar).to_numpy(dtype=float)
    factors = None
    if factors_frame is not None:
        from ashare_event.market import prepare_factors

        prepared = prepare_factors(factors_frame)
        if prepared is not None:
            aligned = prepared.set_index("date").reindex(calendar)
            factors = aligned[["mkt", "smb", "hml"]].to_numpy(dtype=float)
    return _Store(calendar=calendar, mkt=mkt.reindex(calendar).to_numpy(dtype=float), firm=firm, industry=industry, factors=factors)


def _codes(events: pd.DataFrame) -> np.ndarray:
    return events["code"].astype(str).str.zfill(6).str[-6:].to_numpy()


def _industries(events: pd.DataFrame) -> np.ndarray:
    if "industry" not in events.columns:
        return np.array([""] * len(events), dtype=object)
    return events["industry"].astype(str).to_numpy()


def _positions_from_dates(events: pd.DataFrame, store: _Store) -> np.ndarray:
    out = np.full(len(events), -1, dtype=int)
    for i, event in enumerate(events.itertuples(index=False)):
        aligned = _align(
            store.calendar,
            event.event_date,
            after=parse_after_close(getattr(event, "after_close", False)),
        )
        if aligned is not None:
            out[i] = aligned[0]
    return out


def _positions_from_aligned(events: pd.DataFrame, store: _Store) -> np.ndarray:
    if "event_date_aligned" not in events.columns:
        return _positions_from_dates(events, store)
    out = np.full(len(events), -1, dtype=int)
    for i, day in enumerate(pd.to_datetime(events["event_date_aligned"])):
        if pd.isna(day):
            continue
        try:
            out[i] = int(store.calendar.get_loc(pd.Timestamp(day).normalize()))
        except KeyError:
            out[i] = -1
    return out


def _one(store: _Store, code: str, pos: int, industry: str, cfg: StudyConfig) -> float:
    ret = store.firm.get(str(code).zfill(6)[-6:])
    if ret is None:
        return np.nan
    model = cfg.model
    benchmark = store.mkt
    factors = None
    if model == "industry":
        benchmark = store.industry.get(str(industry))
        if benchmark is None:
            return np.nan
        model = "market"
    elif model == "ff3":
        factors = store.factors
        if factors is None:
            return np.nan
    return car_on_position(
        ret,
        benchmark,
        int(pos),
        cfg.estimation_window,
        cfg.regression_window,
        model,
        cfg.min_estimation_obs,
        cfg.require_complete_window,
        factors,
    )


def event_cars(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    market: pd.DataFrame,
    cfg: StudyConfig,
    industry_frame: pd.DataFrame | None = None,
    factors_frame: pd.DataFrame | None = None,
) -> pd.Series:
    """Regression-window CAR for each event, using the same kernel as the placebos."""
    store = _store(prices, market, industry_frame, factors_frame)
    positions = _positions_from_dates(events, store)
    codes = _codes(events)
    industries = _industries(events)
    values = [
        np.nan if positions[i] < 0 else _one(store, codes[i], int(positions[i]), industries[i], cfg)
        for i in range(len(events))
    ]
    return pd.Series(values, index=events["event_id"].to_numpy(), dtype=float)


def _mean_of(values: np.ndarray, attempted: int) -> float:
    finite = values[np.isfinite(values)]
    if attempted <= 0 or finite.size < max(5, int(0.5 * attempted)):
        return np.nan
    return float(finite.mean())


def _p_value(actual: float, draws: np.ndarray) -> float:
    if draws.size == 0 or not np.isfinite(actual):
        return np.nan
    return float((1 + np.sum(np.abs(draws) >= abs(actual))) / (1 + draws.size))


def _percentile(actual: float, draws: np.ndarray) -> float:
    if draws.size == 0 or not np.isfinite(actual):
        return np.nan
    return float(100.0 * np.mean(draws <= actual))


def placebo_sentence(language: str, mode: str, pack: dict, scale: float) -> str:
    if int(pack.get("draws") or 0) == 0 or not np.isfinite(pack.get("percentile", np.nan)):
        if language == "en":
            return "This placebo did not produce enough draws. That is not evidence of an effect."
        return "这次安慰剂没有凑够可用的抽取。这不能当成有效应的证据。"
    actual = float(pack["actual"]) * scale
    pct = float(pack["percentile"])
    p_value = float(pack["p_value"])
    if language == "en":
        return (
            f"The actual mean CAR is {actual:.3f} display units and sits at percentile {pct:.1f} "
            f"of the {mode.replace('_', '-')} placebo draws (two-sided randomization p≈{p_value:.3f}). "
            "This says whether the real date stands out among these pseudo-dates. It is not a p-value for a treatment effect."
        )
    name = "时间安慰剂" if mode == "in_time" else "截面安慰剂"
    return (
        f"观测到的平均 CAR 约为 {actual:.3f} 个百分点，落在{name}分布的第 {pct:.1f} 百分位"
        f"（双侧随机化 p≈{p_value:.3f}）。这只说明真实日期在这批伪日期里是否突出，不是因果效应的 p 值。"
    )


def _pack(mode: str, actual: float, draws: np.ndarray) -> dict:
    if draws.size == 0:
        return {
            "mode": mode,
            "draws": 0,
            "actual": actual,
            "null_mean": np.nan,
            "p05": np.nan,
            "p95": np.nan,
            "percentile": np.nan,
            "p_value": np.nan,
            "nulls": draws,
        }
    return {
        "mode": mode,
        "draws": int(draws.size),
        "actual": float(actual),
        "null_mean": float(draws.mean()),
        "p05": float(np.quantile(draws, 0.05)),
        "p95": float(np.quantile(draws, 0.95)),
        "percentile": _percentile(actual, draws),
        "p_value": _p_value(actual, draws),
        "nulls": draws,
    }


def in_time_placebo(events, prices, market, cfg, actual, rng, industry_frame=None, factors_frame=None, store=None) -> dict:
    store = store or _store(prices, market, industry_frame, factors_frame)
    base = _positions_from_aligned(events, store)
    codes = _codes(events)
    industries = _industries(events)
    n_obs = len(events)
    low, high = cfg.placebo_offsets
    draws = []
    for draw in range(cfg.placebo_draws):
        offsets = rng.integers(int(low), int(high) + 1, size=n_obs)
        pos = base + offsets.astype(int)
        valid = (base >= 0) & (pos >= 0) & (pos < len(store.calendar))
        values = np.full(n_obs, np.nan)
        for i in np.flatnonzero(valid):
            values[i] = _one(store, codes[i], int(pos[i]), industries[i], cfg)
        value = _mean_of(values, int(valid.sum()))
        if np.isfinite(value):
            draws.append(value)
        if cfg.verbose and (draw + 1) % 50 == 0:
            print(f"时间安慰剂 {draw + 1}/{cfg.placebo_draws}", flush=True)
    return _pack("in_time", actual, np.asarray(draws, dtype=float))


def in_space_placebo(events, prices, market, cfg, actual, rng, industry_frame=None, factors_frame=None, store=None) -> dict:
    store = store or _store(prices, market, industry_frame, factors_frame)
    base = _positions_from_aligned(events, store)
    codes = _codes(events)
    industries = _industries(events)
    n_obs = len(events)
    draws = []
    for draw in range(cfg.placebo_draws):
        shuffled = rng.permutation(codes)
        values = np.full(n_obs, np.nan)
        for i in range(n_obs):
            if base[i] < 0:
                continue
            values[i] = _one(store, shuffled[i], int(base[i]), industries[i], cfg)
        value = _mean_of(values, n_obs)
        if np.isfinite(value):
            draws.append(value)
        if cfg.verbose and (draw + 1) % 50 == 0:
            print(f"截面安慰剂 {draw + 1}/{cfg.placebo_draws}", flush=True)
    return _pack("in_space", actual, np.asarray(draws, dtype=float))


def run_placebos(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    market: pd.DataFrame,
    cfg: StudyConfig,
    actual: float,
    rng: np.random.Generator,
    industry_frame: pd.DataFrame | None = None,
    factors_frame: pd.DataFrame | None = None,
) -> dict[str, dict]:
    store = _store(prices, market, industry_frame, factors_frame)
    results = {}
    if "in_time" in cfg.placebo_modes:
        results["in_time"] = in_time_placebo(
            events, prices, market, cfg, actual, rng, store=store
        )
    if "in_space" in cfg.placebo_modes:
        results["in_space"] = in_space_placebo(
            events, prices, market, cfg, actual, rng, store=store
        )
    for mode, pack in results.items():
        pack["sentence"] = placebo_sentence(cfg.language, mode, pack, cfg.return_scale)
    return results
