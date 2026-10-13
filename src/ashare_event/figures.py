"""Paper figures. One claim per figure. Vector PDF, Okabe-Ito, 6.5 by 4 inches.

The CAR-path band is the cross-sectional 95 percent interval of the mean.
When the average residual correlation is positive, the band is widened by
the Kolari-Pynnonen factor so the picture does not look more precise than
the dependence allows.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.getLogger("fontTools").setLevel(logging.ERROR)

from ashare_event.inference import kp_factor
from ashare_event.labels import text

BLUE = "#0072B2"
VERMILLION = "#D55E00"
BAND = "#0072B2"


def _save(fig, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight")
    if destination.suffix.lower() == ".pdf":
        fig.savefig(destination.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _theme() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.unicode_minus": False,
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_car_path(
    path: pd.DataFrame,
    destination: Path,
    *,
    language: str,
    scale: float,
    r_bar: float,
    size: tuple[float, float],
) -> str:
    """Return a one-sentence caption. The figure itself has no title."""
    _theme()
    fig, ax = plt.subplots(figsize=size)
    if path.empty:
        _save(fig, destination)
        return ""
    order = path.sort_values("rel_day")
    x = order["rel_day"].to_numpy()
    mean = order["mean"].to_numpy(dtype=float) * scale
    se = order["se"].to_numpy(dtype=float) * scale
    n_obs = int(order["n"].max()) if len(order) else 0
    factor = kp_factor(r_bar, n_obs)
    widened = np.isfinite(r_bar) and r_bar > 0.02 and factor < 1
    if widened:
        se = se / factor
    band = 1.96 * se
    ax.axhline(0, color="#666666", linewidth=0.6)
    ax.axvline(0, color=VERMILLION, linewidth=0.9)
    ax.plot(x, mean, color=BLUE, linewidth=1.6)
    ok = np.isfinite(band)
    ax.fill_between(x[ok], (mean - band)[ok], (mean + band)[ok], color=BAND, alpha=0.18, linewidth=0)
    ax.set_xlabel(text(language, "rel_day"))
    ax.set_ylabel(text(language, "mean_car"))
    fig.tight_layout()
    _save(fig, destination)
    if language == "zh":
        band_note = "阴影是按平均残差相关加宽后的 95% 区间。" if widened else "阴影是横截面 95% 区间。"
        return f"事件路径上的平均累计异常收益。N = {n_obs}。{band_note}竖线为事件日。"
    band_note = (
        "The band is a 95 percent interval widened by the average residual correlation."
        if widened
        else "The band is a cross-sectional 95 percent interval."
    )
    return f"Mean cumulative abnormal return in event time. N = {n_obs}. {band_note} The vertical line is the event day."


def save_placebo(
    result: dict,
    destination: Path,
    *,
    language: str,
    scale: float,
    size: tuple[float, float],
) -> str:
    _theme()
    fig, ax = plt.subplots(figsize=size)
    nulls = np.asarray(result.get("nulls", []), dtype=float)
    nulls = nulls[np.isfinite(nulls)] * scale
    actual = result.get("actual", np.nan) * scale
    if nulls.size:
        ax.hist(nulls, bins=min(20, max(5, int(np.sqrt(nulls.size)))), color=BLUE, alpha=0.85, edgecolor="white")
    if np.isfinite(actual):
        ax.axvline(actual, color=VERMILLION, linewidth=1.4)
    ax.set_xlabel(text(language, "placebo_x"))
    ax.set_ylabel(text(language, "placebo_y"))
    fig.tight_layout()
    _save(fig, destination)
    name = text(language, result["mode"])
    if language == "zh":
        return f"{name}。柱为安慰剂分布，竖线为实际平均累计异常收益。"
    return f"{name}. Bars are the placebo distribution and the vertical line is the actual mean CAR."


def write_captions(path: Path, captions: dict[str, str]) -> None:
    lines = ["% Captions produced by the pipeline. Edit the paper, not these sentences, if a number changes.", ""]
    for key, caption in captions.items():
        lines.append(f"% {key}")
        lines.append(caption)
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
