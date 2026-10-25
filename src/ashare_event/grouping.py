"""Group means and paths. Stars here are cross-sectional t statistics.

They are not the preferred event-study test, and they are not multiplicity-adjusted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ashare_event.figures import save_group_bars, save_grouped_paths
from ashare_event.inference import mean_t
from ashare_event.labels import event_type_label
from ashare_event.tables import stars


def _label(column: str, name, language: str) -> str:
    if column == "event_type":
        return event_type_label(language, str(name))
    return str(name)


def group_means(events: pd.DataFrame, column: str, car_col: str, scale: float, language: str) -> pd.DataFrame:
    rows = []
    if events.empty or car_col not in events.columns or column not in events.columns:
        return pd.DataFrame(columns=["label", "mean", "se", "stars", "n"])
    for name, part in events.groupby(column, sort=False):
        values = pd.to_numeric(part[car_col], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size < 2:
            continue
        mean, _t, p_value = mean_t(values)
        rows.append(
            {
                "label": _label(column, name, language),
                "mean": mean * scale,
                "se": float(values.std(ddof=1) / np.sqrt(values.size)) * scale,
                "stars": stars(p_value),
                "n": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def group_paths(
    events: pd.DataFrame,
    panel: pd.DataFrame,
    column: str,
    scale: float,
    language: str,
    limit: int = 6,
) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["group", "rel_day", "mean", "se"])
    if events.empty or panel.empty or "car_path" not in panel.columns or column not in events.columns:
        return empty
    keep_names = set(events[column].value_counts().head(limit).index)
    meta = events.loc[events[column].isin(keep_names), ["event_id", column]]
    merged = panel.dropna(subset=["car_path"]).merge(meta, on="event_id", how="inner")
    if merged.empty:
        return empty
    rows = []
    for (group, day), part in merged.groupby([column, "rel_day"], sort=False):
        values = pd.to_numeric(part["car_path"], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size < 2:
            continue
        rows.append(
            {
                "group": _label(column, group, language),
                "rel_day": int(day),
                "mean": float(values.mean()) * scale,
                "se": float(values.std(ddof=1) / np.sqrt(values.size)) * scale,
            }
        )
    return pd.DataFrame(rows)


def build_group_frames(events: pd.DataFrame, panel: pd.DataFrame, car_col: str, scale: float, language: str) -> dict:
    return {
        "industry_bars": group_means(events, "industry", car_col, scale, language),
        "type_bars": group_means(events, "event_type", car_col, scale, language),
        "industry_paths": group_paths(events, panel, "industry", scale, language),
        "type_paths": group_paths(events, panel, "event_type", scale, language),
    }


def write_group_figures(groups: dict, figure_dir: Path, size: tuple[float, float]) -> None:
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    bars = {
        "industry_bars": "fig_car_by_industry.png",
        "type_bars": "fig_car_by_type.png",
    }
    paths = {
        "industry_paths": "fig_car_path_by_industry.png",
        "type_paths": "fig_car_path_by_type.png",
    }
    for key, name in bars.items():
        frame = groups.get(key)
        if frame is None or frame.empty:
            continue
        save_group_bars(frame, figure_dir / name, size=size)
    for key, name in paths.items():
        frame = groups.get(key)
        if frame is None or frame.empty or frame["group"].nunique() <= 1:
            continue
        save_grouped_paths(frame, figure_dir / name, size=size)
