"""Placebos for the mean CAR.

In-time moves each firm to another date of its own history. The null says
this date is not special for these firms.

In-space reassigns price series across the sample. The null says this
firm's return is not special on this date.

Neither placebo is a control group, and neither delivers an ATT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_event.config import StudyConfig, window_suffix
from ashare_event.returns import _returns_and_calendar, estimate_events


def _kwargs(cfg: StudyConfig) -> dict:
    return {
        "estimation_window": cfg.estimation_window,
        "event_windows": cfg.event_windows,
        "path_window": cfg.path_window,
        "regression_window": cfg.regression_window,
        "model": cfg.model,
        "min_estimation_obs": cfg.min_estimation_obs,
        "require_complete_window": cfg.require_complete_window,
    }


def _p_value(actual: float, draws: np.ndarray) -> float:
    if draws.size == 0 or not np.isfinite(actual):
        return np.nan
    return float((1 + np.sum(np.abs(draws) >= abs(actual))) / (1 + draws.size))


def _mean_car(events: pd.DataFrame, prices, market, cfg: StudyConfig, suffix: str) -> float:
    estimated = estimate_events(events, prices, market, **_kwargs(cfg))
    column = f"car_{suffix}"
    if estimated.events.empty or column not in estimated.events.columns:
        return np.nan
    values = estimated.events[column].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size < max(5, int(0.5 * len(events))):
        return np.nan
    return float(values.mean())


def in_time_placebo(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    market: pd.DataFrame,
    cfg: StudyConfig,
    actual: float,
    rng: np.random.Generator,
) -> dict:
    _, _, calendar = _returns_and_calendar(prices, market)
    suffix = window_suffix(*cfg.regression_window)
    low, high = cfg.placebo_offsets
    draws = []
    aligned = pd.to_datetime(events["event_date_aligned"]).dt.normalize()
    positions = [int(calendar.get_loc(day)) for day in aligned]
    for draw in range(cfg.placebo_draws):
        offsets = rng.integers(low, high + 1, size=len(events))
        new_dates = []
        for pos, offset in zip(positions, offsets, strict=True):
            new_pos = int(pos) + int(offset)
            if new_pos < 0 or new_pos >= len(calendar):
                new_dates.append(pd.NaT)
            else:
                new_dates.append(calendar[new_pos])
        shifted = events.copy()
        shifted["event_date"] = new_dates
        shifted = shifted.dropna(subset=["event_date"])
        if shifted.empty:
            continue
        value = _mean_car(shifted, prices, market, cfg, suffix)
        if np.isfinite(value):
            draws.append(value)
        if cfg.verbose and (draw + 1) % 50 == 0:
            print(f"时间安慰剂 {draw + 1}/{cfg.placebo_draws}", flush=True)
    array = np.asarray(draws, dtype=float)
    return _pack("in_time", actual, array)


def in_space_placebo(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    market: pd.DataFrame,
    cfg: StudyConfig,
    actual: float,
    rng: np.random.Generator,
) -> dict:
    suffix = window_suffix(*cfg.regression_window)
    draws = []
    codes = events["code"].to_numpy().copy()
    for draw in range(cfg.placebo_draws):
        shuffled = events.copy()
        shuffled["code"] = rng.permutation(codes)
        value = _mean_car(shuffled, prices, market, cfg, suffix)
        if np.isfinite(value):
            draws.append(value)
        if cfg.verbose and (draw + 1) % 50 == 0:
            print(f"截面安慰剂 {draw + 1}/{cfg.placebo_draws}", flush=True)
    return _pack("in_space", actual, np.asarray(draws, dtype=float))


def _pack(mode: str, actual: float, draws: np.ndarray) -> dict:
    if draws.size == 0:
        return {
            "mode": mode,
            "draws": 0,
            "actual": actual,
            "null_mean": np.nan,
            "p05": np.nan,
            "p95": np.nan,
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
        "p_value": _p_value(actual, draws),
        "nulls": draws,
    }


def run_placebos(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    market: pd.DataFrame,
    cfg: StudyConfig,
    actual: float,
    rng: np.random.Generator,
) -> dict[str, dict]:
    results = {}
    if "in_time" in cfg.placebo_modes:
        results["in_time"] = in_time_placebo(events, prices, market, cfg, actual, rng)
    if "in_space" in cfg.placebo_modes:
        results["in_space"] = in_space_placebo(events, prices, market, cfg, actual, rng)
    return results
