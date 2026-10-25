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

BLUE = "#1f4e79"
VERMILLION = "#9b3a3a"
BAND = "#1f4e79"


def _save(fig, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    png = destination.with_suffix(".png") if destination.suffix.lower() == ".pdf" else destination
    fig.savefig(png, bbox_inches="tight", dpi=150)
    if destination.suffix.lower() == ".pdf":
        try:
            fig.savefig(destination, bbox_inches="tight")
        except Exception:
            logging.getLogger("ashare_event.figures").warning("PDF 导出失败，已保留 PNG。", exc_info=True)
    plt.close(fig)


def _cjk_family() -> str:
    """Family name of a font that can draw Chinese labels, if one is installed."""
    from matplotlib import font_manager

    preferred = (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
    )
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            return name
    roots = [Path("/usr/share/fonts"), Path("/usr/local/share/fonts")]
    for root in roots:
        if not root.exists():
            continue
        for pattern in ("NotoSansCJK*.ttc", "NotoSansCJK*.otf", "NotoSansSC*.otf", "NotoSansSC*.ttf"):
            matches = sorted(root.rglob(pattern))
            if not matches:
                continue
            try:
                font_manager.fontManager.addfont(str(matches[0]))
            except (RuntimeError, OSError, ValueError):
                continue
            return font_manager.FontProperties(fname=str(matches[0])).get_name()
    return "DejaVu Sans"


def _theme() -> None:
    family = _cjk_family()
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.unicode_minus": False,
            "font.family": "sans-serif",
            "font.sans-serif": [
                family,
                "Noto Sans CJK SC",
                "Noto Sans CJK JP",
                "Noto Sans SC",
                "Microsoft YaHei",
                "SimHei",
                "DejaVu Sans",
            ],
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
    _save(fig, destination)
    name = text(language, result["mode"])
    if language == "zh":
        return f"{name}。柱为安慰剂分布，竖线为实际平均累计异常收益。"
    return f"{name}. Bars are the placebo distribution and the vertical line is the actual mean CAR."


def save_group_bars(
    frame: pd.DataFrame,
    destination: Path,
    *,
    size: tuple[float, float],
) -> None:
    """Horizontal mean-CAR bars. The caption lives outside the figure."""
    _theme()
    fig, ax = plt.subplots(figsize=size)
    if frame.empty:
        _save(fig, destination)
        return
    ordered = frame.sort_values("mean")
    colors = ["#1f4e79" if value >= 0 else "#9b3a3a" for value in ordered["mean"]]
    y = np.arange(len(ordered))
    ax.barh(y, ordered["mean"], color=colors, xerr=1.96 * ordered["se"], ecolor="#8aa0b8", capsize=2)
    ax.axvline(0, color="#8aa0b8", linewidth=0.7)
    labels = [f"{name}{stars}" for name, stars in zip(ordered["label"], ordered["stars"], strict=True)]
    ax.set_yticks(y, labels)
    ax.set_xlabel("平均 CAR")
    _save(fig, destination)


GROUP_COLORS = ("#1f4e79", "#1d6b45", "#9b3a3a", "#5a6a7a", "#8aa0b8", "#3d4c5c")


def save_grouped_paths(frame: pd.DataFrame, destination: Path, *, size: tuple[float, float]) -> None:
    """One cumulative-CAR path per group, each with a standard-error band."""
    _theme()
    fig, ax = plt.subplots(figsize=size)
    if frame.empty:
        _save(fig, destination)
        return
    ax.axhline(0, color="#8aa0b8", linewidth=0.6)
    ax.axvline(0, color="#9b3a3a", linewidth=0.9)
    for index, (name, part) in enumerate(frame.groupby("group", sort=False)):
        color = GROUP_COLORS[index % len(GROUP_COLORS)]
        order = part.sort_values("rel_day")
        x_values = order["rel_day"].to_numpy()
        mean = order["mean"].to_numpy(dtype=float)
        band = 1.96 * order["se"].to_numpy(dtype=float)
        ok = np.isfinite(mean) & np.isfinite(band)
        ax.plot(x_values[np.isfinite(mean)], mean[np.isfinite(mean)], color=color, linewidth=1.5, label=str(name))
        ax.fill_between(x_values[ok], (mean - band)[ok], (mean + band)[ok], color=color, alpha=0.12, linewidth=0)
    ax.set_xlabel("相对事件日的交易日")
    ax.set_ylabel("平均 CAR")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8, loc="best")
    _save(fig, destination)


def _plotly():
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    return go


def _plotly_layout(fig):
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin={"l": 48, "r": 20, "t": 24, "b": 48},
        font={"family": "Microsoft YaHei, Noto Sans SC, sans-serif", "size": 13},
        legend={"orientation": "h", "y": 1.12},
    )
    return fig


def plotly_car_path(path: pd.DataFrame, scale: float, r_bar: float):
    go = _plotly()
    if go is None or path is None or path.empty:
        return None
    order = path.sort_values("rel_day")
    mean = order["mean"].to_numpy(dtype=float) * scale
    se = order["se"].to_numpy(dtype=float) * scale
    n_obs = int(order["n"].max()) if len(order) else 0
    factor = kp_factor(r_bar, n_obs)
    if np.isfinite(r_bar) and r_bar > 0.02 and factor < 1:
        se = se / factor
    band = 1.96 * se
    x = order["rel_day"].to_numpy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(x) + list(x[::-1]),
            y=list(mean + band) + list((mean - band)[::-1]),
            fill="toself",
            fillcolor="rgba(31,78,121,0.15)",
            line={"width": 0},
            hoverinfo="skip",
            name="95% 区间",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines",
            name="平均 CAR",
            line={"color": "#1f4e79", "width": 2},
            customdata=order["n"],
            hovertemplate="相对日 %{x}<br>平均 CAR %{y:.3f}<br>N %{customdata}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color="#9b3a3a", line_width=1)
    fig.update_xaxes(title="相对事件日的交易日")
    fig.update_yaxes(title="平均 CAR（百分点）")
    return _plotly_layout(fig)


def plotly_grouped_paths(frame: pd.DataFrame):
    go = _plotly()
    if go is None or frame is None or frame.empty:
        return None
    fig = go.Figure()
    fig.add_vline(x=0, line_color="#9b3a3a", line_width=1)
    for name, part in frame.groupby("group", sort=False):
        order = part.sort_values("rel_day")
        fig.add_trace(
            go.Scatter(
                x=order["rel_day"],
                y=order["mean"],
                mode="lines",
                name=str(name),
                hovertemplate="相对日 %{x}<br>" + str(name) + " %{y:.3f}<extra></extra>",
            )
        )
    fig.update_xaxes(title="相对事件日的交易日")
    fig.update_yaxes(title="平均 CAR（百分点）")
    return _plotly_layout(fig)


def plotly_group_bars(frame: pd.DataFrame):
    go = _plotly()
    if go is None or frame is None or frame.empty:
        return None
    ordered = frame.sort_values("mean")
    fig = go.Figure(
        go.Bar(
            x=ordered["mean"],
            y=ordered["label"],
            orientation="h",
            marker_color=["#1f4e79" if value >= 0 else "#9b3a3a" for value in ordered["mean"]],
            customdata=np.column_stack([ordered["n"], ordered["se"]]),
            hovertemplate="%{y}<br>平均 CAR %{x:.3f}<br>N %{customdata[0]}<extra></extra>",
            error_x={"type": "data", "array": 1.96 * ordered["se"], "color": "#8aa0b8"},
        )
    )
    fig.update_xaxes(title="平均 CAR（百分点）")
    return _plotly_layout(fig)


def plotly_placebo(result: dict, scale: float):
    go = _plotly()
    if go is None:
        return None
    nulls = np.asarray(result.get("nulls", []), dtype=float)
    nulls = nulls[np.isfinite(nulls)] * scale
    if nulls.size == 0:
        return None
    fig = go.Figure(go.Histogram(x=nulls, marker_color="#1f4e79", name="安慰剂"))
    actual = result.get("actual", np.nan)
    if np.isfinite(actual):
        fig.add_vline(x=float(actual) * scale, line_color="#9b3a3a", line_width=2)
    fig.update_xaxes(title="安慰剂平均 CAR（百分点）")
    fig.update_yaxes(title="抽取次数")
    return _plotly_layout(fig)


def write_captions(path: Path, captions: dict[str, str]) -> None:
    lines = ["% Captions produced by the pipeline. Edit the paper, not these sentences, if a number changes.", ""]
    for key, caption in captions.items():
        lines.append(f"% {key}")
        lines.append(caption)
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
