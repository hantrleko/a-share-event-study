"""One zip of the files that belong to a single run."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def build_zip(result) -> bytes:
    cfg = result.config
    outputs = result.outputs or {}
    figure_dir = Path(outputs["figures"]) if outputs.get("figures") else None
    table_dir = Path(outputs["tables"]) if outputs.get("tables") else None
    windows = ", ".join(f"[{int(row.t1)}, {int(row.t2)}]" for row in result.tests.itertuples(index=False)) if len(result.tests) else ""
    firms = int(result.events["firm_id"].nunique()) if len(result.events) and "firm_id" in result.events.columns else 0
    lines = [
        "A股事件研究 · 本次运行说明",
        "",
        f"版本：{getattr(result, 'manifest', {}).get('app_version', '')}",
        f"运行编号：{getattr(result, 'manifest', {}).get('run_id', '')}",
        f"模式：{cfg.mode}",
        f"收益模型：{cfg.model}",
        f"基准：{cfg.benchmark}",
        f"复权：{cfg.adjust or '不复权'}",
        f"估计窗：[{cfg.estimation_window[0]}, {cfg.estimation_window[1]}]",
        f"路径窗：[{cfg.path_window[0]}, {cfg.path_window[1]}]",
        f"回归窗：[{cfg.regression_window[0]}, {cfg.regression_window[1]}]",
        f"CAR 窗：{windows}",
        f"随机种子：{cfg.seed}",
        f"进入估计的事件数：{len(result.events)}",
        f"公司数：{firms}",
        f"事件表摘要：{getattr(result, 'manifest', {}).get('events_hash', '')}",
        "",
        "市场模型异常收益是相对基准的统计偏离，不是平均处理效应。",
        "时间安慰剂会在伪事件日上重新拟合估计窗。",
        "星号来自首选检验，没有做多重检验校正。p BH 只调整窗口表里的首选 p 值。",
        "方法段见 methods_zh.md 与 methods_en.md。完整参数见 manifest.json。",
    ]
    if result.simulated:
        lines.append("本次数值来自模拟样本，不能写入论文。")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("methods_note.txt", ("\n".join(lines) + "\n").encode("utf-8"))
        if getattr(result, "methods_zh", ""):
            archive.writestr("methods_zh.md", result.methods_zh.encode("utf-8"))
        if getattr(result, "methods_en", ""):
            archive.writestr("methods_en.md", result.methods_en.encode("utf-8"))
        if getattr(result, "manifest", None):
            archive.writestr("manifest.json", json.dumps(result.manifest, ensure_ascii=False, indent=2).encode("utf-8"))
        if getattr(result, "sample_log", ""):
            archive.writestr("sample_log.md", result.sample_log.encode("utf-8"))
        archive.writestr("csv/car_windows.csv", result.tests.to_csv(index=False).encode("utf-8-sig"))
        archive.writestr("csv/event_car.csv", result.events.to_csv(index=False).encode("utf-8-sig"))
        vif = getattr(result, "vif", None)
        if vif is not None and len(getattr(vif, "index", [])):
            archive.writestr("csv/vif.csv", vif.to_csv(index=False).encode("utf-8-sig"))
        dictionary = getattr(result, "dictionary", None)
        if dictionary is not None and len(getattr(dictionary, "index", [])):
            archive.writestr("csv/variable_dictionary.csv", dictionary.to_csv(index=False).encode("utf-8-sig"))
        for mode, pack in (result.placebos or {}).items():
            nulls = np.asarray(pack.get("nulls", []), dtype=float)
            if nulls.size:
                frame = pd.DataFrame({"null_mean_car": nulls})
                archive.writestr(f"csv/placebo_{mode}.csv", frame.to_csv(index=False).encode("utf-8-sig"))
        if table_dir is not None and table_dir.exists():
            for path in sorted(table_dir.glob("*.tex")):
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    archive.writestr(f"tex/{path.name}", text.encode("utf-8"))
        if figure_dir is not None and figure_dir.exists():
            for path in sorted(figure_dir.glob("*.png")):
                if path.stat().st_size >= 800:
                    archive.writestr(f"png/{path.name}", path.read_bytes())
    return buf.getvalue()
