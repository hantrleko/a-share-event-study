"""A short grid of alternative specifications.

This is not part of every study run. The preferred test in the main table
stays the one from the specification the user actually chose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_event.config import ConfigError, StudyConfig, clone_config, validate_windows, window_suffix
from ashare_event.inference import mean_residual_correlation, summarize_window
from ashare_event.returns import estimate_events, winsorize_events


def _specs(cfg: StudyConfig) -> list[dict]:
    specs: list[dict] = []
    seen: set[tuple] = set()

    def add(model: str, estimation: tuple[int, int], winsor: float, label: str) -> None:
        if len(specs) >= 4:
            return
        key = (model, tuple(estimation), round(float(winsor), 4))
        if key in seen:
            return
        seen.add(key)
        specs.append(
            {
                "label": label,
                "model": model,
                "estimation": (int(estimation[0]), int(estimation[1])),
                "winsor": float(winsor),
            }
        )

    add(cfg.model, cfg.estimation_window, 0.0, "当前设定")
    add("market_adjusted" if cfg.model == "market" else "market", cfg.estimation_window, 0.0, "对照模型")
    add(cfg.model, cfg.estimation_window, 0.01, "上下 1% 缩尾")
    left = -180 if cfg.estimation_window[0] < -180 else cfg.estimation_window[0] + 50
    shorter = (int(left), int(cfg.estimation_window[1]))
    probe = clone_config(cfg)
    probe.estimation_window = shorter
    probe.winsorize = 0.0
    if probe.anticipation_window is not None and probe.anticipation_window[0] <= shorter[1]:
        probe.anticipation_window = None
    try:
        validate_windows(probe)
    except ConfigError:
        shorter = None
    if shorter is not None:
        add(cfg.model, shorter, 0.0, "较短估计窗")
    return specs


def _preferred_t(row: dict) -> float:
    name = {"portfolio": "t_portfolio", "kp": "t_kp", "bmp": "t_bmp"}.get(row.get("preferred"), "")
    value = row.get(name, np.nan)
    return float(value) if value is not None else np.nan


def run_robustness(
    cfg: StudyConfig,
    *,
    industries: list[str] | None = None,
    event_types: list[str] | None = None,
    events_frame: pd.DataFrame | None = None,
    prices_frame: pd.DataFrame | None = None,
    market_frame: pd.DataFrame | None = None,
    industry_frame: pd.DataFrame | None = None,
    factors_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Re-estimate a few specs. No placebo and no cross-sectional regression."""
    from ashare_event.pipeline import _load_inputs

    cfg = clone_config(cfg)
    rng = np.random.default_rng(cfg.seed)
    filtered, prices, market, _simulated, _loaded, _screened = _load_inputs(
        cfg,
        rng,
        industries,
        event_types,
        events_frame=events_frame,
        prices_frame=prices_frame,
        market_frame=market_frame,
    )
    rows = []
    for spec in _specs(cfg):
        row = {
            "label": spec["label"],
            "model": spec["model"],
            "estimation": f"[{spec['estimation'][0]}, {spec['estimation'][1]}]",
            "winsor": spec["winsor"],
            "n": 0,
            "mean_car": np.nan,
            "preferred": "",
            "t_preferred": np.nan,
            "mean_bhar": np.nan,
            "note": "",
        }
        try:
            estimated = estimate_events(
                filtered,
                prices,
                market,
                estimation_window=spec["estimation"],
                event_windows=list(cfg.event_windows),
                path_window=cfg.path_window,
                regression_window=cfg.regression_window,
                model=spec["model"],
                min_estimation_obs=cfg.min_estimation_obs,
                require_complete_window=cfg.require_complete_window,
                industry_frame=industry_frame if spec["model"] in {"industry", cfg.model} else None,
                factors_frame=factors_frame if spec["model"] in {"ff3", "ff5", "carhart", cfg.model} else None,
                market_lags=int(cfg.market_lags or 0) if spec["model"] == cfg.model else 0,
            )
        except (ValueError, KeyError) as exc:
            row["note"] = str(exc)
            rows.append(row)
            continue
        events = estimated.events
        if events.empty:
            row["note"] = "这一格没有事件留下。"
            rows.append(row)
            continue
        if spec["winsor"] > 0:
            events = winsorize_events(events, spec["winsor"])
        suffix = window_suffix(*cfg.regression_window)
        summary = summarize_window(
            events,
            estimated.panel,
            cfg.regression_window,
            suffix,
            estimation_window=spec["estimation"],
            model=spec["model"],
            r_bar=mean_residual_correlation(estimated.panel, events["event_id"], cfg.rbar_min_periods),
            min_share=cfg.portfolio_min_share,
        )
        bhar_col = f"bhar_{suffix}"
        row.update(
            {
                "n": int(summary["n"]),
                "mean_car": float(summary["mean"]) if np.isfinite(summary["mean"]) else np.nan,
                "preferred": summary["preferred"],
                "t_preferred": _preferred_t(summary),
                "mean_bhar": float(events[bhar_col].mean()) if bhar_col in events.columns else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
