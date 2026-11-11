"""A log of windows, models, and groups that were actually run.

The log is an audit trail. Stars stay on each run's own unadjusted preferred p.
The log is not a BH or Holm adjustment across the N trials.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ashare_event.tables import _num, _wrap_table, tex_escape

SPEC_NOTE = "这张表只登记已经跑完的窗口、模型和分组。星号仍来自每一次自己的首选 p 值，没有按这 N 次再做 BH 或 Holm 校正。"

MODEL_ZH = {
    "market": "市场模型",
    "market_adjusted": "市场调整",
    "industry": "行业基准",
    "ff3": "三因子",
    "ff5": "五因子",
    "carhart": "Carhart",
}
PREFERRED_ZH = {"portfolio": "组合", "kp": "KP", "bmp": "BMP"}
SPEC_LABELS = {
    "trial": "次序",
    "ran_at": "时间",
    "source": "来源",
    "model": "模型",
    "benchmark": "基准",
    "estimation": "估计窗",
    "car_window": "CAR 窗",
    "event_windows": "全部窗口",
    "grouping": "分组",
    "winsorize": "缩尾",
    "market_lags": "市场领先滞后",
    "drop_limit": "剔除涨跌停",
    "n": "事件数",
    "mean_car_pp": "平均 CAR（百分点）",
    "preferred": "首选检验",
    "t_preferred": "首选 t",
    "simulated": "模拟",
}
DISPLAY_COLUMNS = (
    "trial",
    "ran_at",
    "source",
    "model",
    "car_window",
    "grouping",
    "n",
    "mean_car_pp",
    "preferred",
    "t_preferred",
    "simulated",
)


def spec_row(result, *, source: str, industries=None, event_types=None) -> dict:
    """One completed run. Does not recompute CAR."""
    cfg = result.config
    window = (int(cfg.regression_window[0]), int(cfg.regression_window[1]))
    matched = result.tests[(result.tests["t1"] == window[0]) & (result.tests["t2"] == window[1])] if len(result.tests) else result.tests
    row = None if matched.empty else matched.iloc[0]
    preferred = "" if row is None else str(row["preferred"])
    t_value = None
    if row is not None and preferred in {"portfolio", "kp", "bmp"}:
        t_value = _finite(row[f"t_{preferred}"])
    mean_pp = None if row is None else _finite(float(row["mean"]) * float(cfg.return_scale))
    windows = ", ".join(f"[{int(left)}, {int(right)}]" for left, right in cfg.event_windows)
    return {
        "ran_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "model": cfg.model,
        "benchmark": cfg.benchmark,
        "estimation": f"[{int(cfg.estimation_window[0])}, {int(cfg.estimation_window[1])}]",
        "car_window": f"[{window[0]}, {window[1]}]",
        "event_windows": windows,
        "grouping": grouping_text(industries, event_types),
        "winsorize": float(cfg.winsorize or 0),
        "market_lags": int(cfg.market_lags or 0),
        "drop_limit": bool(cfg.drop_event_day_limit),
        "n": None if row is None else int(row["n"]),
        "mean_car_pp": mean_pp,
        "preferred": preferred,
        "t_preferred": t_value,
        "simulated": bool(getattr(result, "simulated", False)),
    }


def spec_table(rows) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return frame
    frame.insert(0, "trial", np.arange(1, len(frame) + 1))
    return frame.reset_index(drop=True)


def spec_sentence(frame: pd.DataFrame | None) -> str:
    n_trials = 0 if frame is None else len(frame)
    return f"共试了 {n_trials} 种规格。{SPEC_NOTE}"


def spec_display(frame: pd.DataFrame) -> pd.DataFrame:
    shown = frame.loc[:, [column for column in DISPLAY_COLUMNS if column in frame.columns]].copy()
    if "model" in shown.columns:
        shown["model"] = shown["model"].map(lambda value: MODEL_ZH.get(str(value), value))
    if "preferred" in shown.columns:
        shown["preferred"] = shown["preferred"].map(lambda value: PREFERRED_ZH.get(str(value), value))
    if "simulated" in shown.columns:
        shown["simulated"] = shown["simulated"].map(lambda value: "是" if bool(value) else "否")
    if "mean_car_pp" in shown.columns:
        shown["mean_car_pp"] = pd.to_numeric(shown["mean_car_pp"], errors="coerce").round(3)
    if "t_preferred" in shown.columns:
        shown["t_preferred"] = pd.to_numeric(shown["t_preferred"], errors="coerce").round(2)
    return shown.rename(columns=SPEC_LABELS)


def spec_tex(frame: pd.DataFrame) -> str:
    header = " & ".join(["次序", "模型", "CAR 窗", "分组", "N", "平均 CAR", "首选", "t"])
    rows = [header + r" \\"]
    for row in frame.itertuples(index=False):
        cells = [
            str(int(row.trial)),
            tex_escape(MODEL_ZH.get(str(row.model), row.model)),
            tex_escape(row.car_window),
            tex_escape(row.grouping),
            "--" if pd.isna(row.n) else str(int(row.n)),
            _num(_finite(row.mean_car_pp), 3),
            tex_escape(PREFERRED_ZH.get(str(row.preferred), row.preferred)),
            _num(_finite(row.t_preferred), 2),
        ]
        rows.append(" & ".join(cells) + r" \\")
    tabular = "\n".join(
        [
            r"\begin{tabular}{lllrlllr}",
            r"\toprule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    notes = [spec_sentence(frame)]
    if "simulated" in frame.columns and frame["simulated"].astype(bool).any():
        notes.append("模拟样本的行不能写入论文。")
    return _wrap_table("已运行规格", "tab:specs", tabular, notes)


def grouping_text(industries, event_types) -> str:
    industry = _join(industries)
    event = _join(event_types)
    return f"行业 {industry or '全部'}；类型 {event or '全部'}"


def persist_specs(result) -> None:
    """Write the log next to the other outputs of this run, when those folders exist."""
    specs = getattr(result, "specs", None)
    if not isinstance(specs, pd.DataFrame) or specs.empty:
        return
    outputs = getattr(result, "outputs", None) or {}
    processed = outputs.get("processed")
    tables = outputs.get("tables")
    if processed:
        folder = Path(processed)
        folder.mkdir(parents=True, exist_ok=True)
        specs.to_csv(folder / "specs_tried.csv", index=False)
    if tables:
        folder = Path(tables)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "table_specs.tex").write_text(spec_tex(specs), encoding="utf-8")


def _join(values) -> str:
    if values is None:
        return ""
    items = []
    for item in values:
        text = str(item).strip()
        if text and text.lower() != "nan":
            items.append(text)
    return "、".join(items)


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number
