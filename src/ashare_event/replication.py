"""Files that describe one run without copying price files or secrets."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import yaml

from ashare_event.config import StudyConfig

_DROP = {
    "alpha",
    "beta",
    "sigma2",
    "n_est",
    "event_date_aligned",
    "rolled_forward",
    "path_complete",
    "limit_band",
    "limit_hit",
    "limit_hit_days",
    "limit_capped_end",
    "year",
    "peer_count_pm5",
    "simulated",
}


def _config_payload(cfg: StudyConfig) -> dict:
    return {
        "project_name": cfg.project_name,
        "seed": int(cfg.seed),
        "mode": cfg.mode,
        "language": cfg.language,
        "model": cfg.model,
        "benchmark": cfg.benchmark,
        "adjust": cfg.adjust,
        "estimation_window": list(cfg.estimation_window),
        "event_windows": [list(item) for item in cfg.event_windows],
        "path_window": list(cfg.path_window),
        "regression_window": list(cfg.regression_window),
        "anticipation_window": None if cfg.anticipation_window is None else list(cfg.anticipation_window),
        "winsorize": float(cfg.winsorize or 0),
        "market_lags": int(cfg.market_lags or 0),
        "min_estimation_obs": int(cfg.min_estimation_obs),
        "require_complete_window": bool(cfg.require_complete_window),
        "cluster": cfg.cluster,
        "two_way_cluster": bool(cfg.two_way_cluster),
        "wild_draws": int(cfg.wild_draws),
        "covariates": list(cfg.covariates),
        "include_peer_count": bool(cfg.include_peer_count),
        "return_scale": float(cfg.return_scale),
        "placebo_draws": int(cfg.placebo_draws),
        "placebo_offsets": list(cfg.placebo_offsets),
        "placebo_modes": list(cfg.placebo_modes),
        "rbar_min_periods": int(cfg.rbar_min_periods),
        "portfolio_min_share": float(cfg.portfolio_min_share),
        "figure_size": list(cfg.figure_size),
        "events_file": "events.csv",
        "prices_file": None,
        "market_file": None,
        "paths": {
            "tables": "replay/tables",
            "figures": "replay/figures",
            "processed": "replay/processed",
            "logs": "replay/logs",
            "theory": "replay/theory",
        },
    }


def _input_events(events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in events.columns
        if column not in _DROP and not str(column).startswith(("car_", "var_", "scar_", "bhar_", "beta_"))
    ]
    return events.loc[:, columns].copy()


def _runner_text() -> str:
    return "\n".join(
        [
            '"""Replay this folder.',
            "",
            "Demo mode rebuilds prices from the seed. This folder does not contain prices.",
            "Local mode needs prices_file and market_file filled in by you.",
            "AKShare mode downloads again and refuses rows whose names still contain 示例.",
            '"""',
            "",
            "from pathlib import Path",
            "",
            "from ashare_event.config import load_config",
            "from ashare_event.pipeline import run_study",
            "",
            'ROOT = Path(__file__).resolve().parent',
            'cfg = load_config(ROOT / "study_repro.yml")',
            'if cfg.mode != "demo":',
            '    raise SystemExit("这份快照没有打包行情。demo 用种子重放；local 请填写价格文件；akshare 会重新下载。")',
            'run_study(cfg, export=True, output_dir=ROOT / "replay")',
            "",
        ]
    )


def _do_text(outcome: str) -> str:
    return "\n".join(
        [
            "* 复现备忘。reghdfe 需要使用者自行安装，本文件不替你安装。",
            "* 系数是累计异常收益和公司特征的相关，不是因果效应。",
            "* 市场模型异常收益不是平均处理效应。",
            'use "event_car.dta", clear',
            f"* 回归窗的累计异常收益列是 {outcome}。按你的协变量改下一行。",
            f'* reghdfe {outcome} signal_cost size_log legitimacy_gap, absorb(industry) vce(cluster firm_id)',
            "",
        ]
    )


def _stata_frame(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    for column in list(frame.columns):
        series = frame[column]
        if pd.api.types.is_datetime64_any_dtype(series) or "date" in str(column).lower():
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().any():
                frame[column] = parsed.dt.strftime("%Y-%m-%d")
                continue
        if pd.api.types.is_bool_dtype(series):
            frame[column] = series.fillna(False).astype(int)
    used: set[str] = set()
    rename = {}
    for column in frame.columns:
        raw = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(column))
        raw = (raw[:28] or "v").strip("_") or "v"
        name = raw
        counter = 1
        while name in used:
            name = f"{raw[:24]}_{counter}"
            counter += 1
        used.add(name)
        rename[column] = name
    frame = frame.rename(columns=rename)
    keep = []
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_string_dtype(series) or series.dtype == object:
            keep.append(column)
    return frame.loc[:, keep]


def stata_bytes(events: pd.DataFrame) -> tuple[bytes | None, str]:
    """Return dta bytes, or a note when Stata export fails. Never raises."""
    try:
        frame = _stata_frame(events)
        buffer = io.BytesIO()
        frame.to_stata(buffer, write_index=False, version=118)
        return buffer.getvalue(), ""
    except Exception as exc:
        return None, f"Stata 文件没有写出：{exc}\n这次运行的 csv 仍然可用。\n"


def replication_files(result) -> dict[str, bytes]:
    cfg = result.config
    payload = yaml.safe_dump(_config_payload(cfg), allow_unicode=True, sort_keys=False)
    events = _input_events(result.events)
    outcome = "car_" + _suffix(cfg)
    files = {
        "replication/study_repro.yml": payload.encode("utf-8"),
        "replication/run_replication.py": _runner_text().encode("utf-8"),
        "replication/events.csv": events.to_csv(index=False).encode("utf-8-sig"),
        "replication/replicate_regressions.do": _do_text(outcome).encode("utf-8"),
        "replication/README.txt": (
            "行情文件不在这个文件夹里。\n"
            "demo 会按 study_repro.yml 里的种子重放模拟样本。\n"
            "local 和 akshare 需要你自己提供价格，或在配置里改回下载。\n"
            "reghdfe 需自行安装。系数是相关，不是因果。\n"
        ).encode("utf-8"),
    }
    data, note = stata_bytes(result.events)
    if data is not None:
        files["replication/event_car.dta"] = data
    else:
        files["replication/stata_note.txt"] = note.encode("utf-8")
    robust = getattr(result, "robustness", None)
    if robust is not None and len(getattr(robust, "index", [])):
        files["replication/robustness.csv"] = robust.to_csv(index=False).encode("utf-8-sig")
    return files


def _suffix(cfg: StudyConfig) -> str:
    from ashare_event.config import window_suffix

    return window_suffix(*cfg.regression_window)


def write_replication(result, dest: str | Path) -> None:
    root = Path(dest)
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in replication_files(result).items():
        (root / Path(name).name).write_bytes(payload)
