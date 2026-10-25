"""Run manifest. Enough to see what was estimated, not a substitute for the input files."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd

from ashare_event import __version__


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return ""


def _git_head(root) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _events_hash(events: pd.DataFrame) -> str:
    columns = [column for column in ("event_id", "code", "event_date", "event_type", "industry") if column in events.columns]
    payload = events[columns].copy() if columns else events.head(0).copy()
    if "event_date" in payload.columns:
        payload["event_date"] = pd.to_datetime(payload["event_date"], errors="coerce").astype(str)
    raw = payload.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _num(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def build_manifest(result) -> dict:
    cfg = result.config
    primary = result.tests
    if not result.tests.empty:
        matched = result.tests[
            (result.tests["t1"] == cfg.regression_window[0]) & (result.tests["t2"] == cfg.regression_window[1])
        ]
        if not matched.empty:
            primary = matched.iloc[0]
        else:
            primary = result.tests.iloc[0]
    else:
        primary = None
    placebos = {}
    for mode, pack in (result.placebos or {}).items():
        placebos[mode] = {
            "draws": int(pack.get("draws") or 0),
            "p_value": _num(pack.get("p_value")),
            "percentile": _num(pack.get("percentile")),
            "sentence": pack.get("sentence", ""),
        }
    vif_max = None
    vif = getattr(result, "vif", None)
    if vif is not None and len(getattr(vif, "index", [])) and "vif" in vif.columns:
        finite = pd.to_numeric(vif["vif"], errors="coerce").dropna()
        if len(finite):
            vif_max = float(finite.max())
    return {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "app_version": __version__,
        "python": sys.version.split()[0],
        "git": _git_head(cfg.root),
        "seed": int(cfg.seed),
        "mode": cfg.mode,
        "simulated": bool(result.simulated),
        "model": cfg.model,
        "benchmark": cfg.benchmark,
        "adjust": cfg.adjust,
        "estimation_window": list(cfg.estimation_window),
        "event_windows": [list(item) for item in cfg.event_windows],
        "path_window": list(cfg.path_window),
        "regression_window": list(cfg.regression_window),
        "cluster": cfg.cluster,
        "two_way_cluster": bool(cfg.two_way_cluster),
        "wild_draws": int(cfg.wild_draws),
        "covariates": list(cfg.covariates),
        "placebo_draws": int(cfg.placebo_draws),
        "placebo_modes": list(cfg.placebo_modes),
        "placebo_offsets": list(cfg.placebo_offsets),
        "n_events": int(len(result.events)),
        "n_firms": int(result.events["firm_id"].nunique()) if len(result.events) and "firm_id" in result.events.columns else 0,
        "events_hash": _events_hash(result.events),
        "r_bar": _num(result.r_bar),
        "preferred": None if primary is None else str(primary["preferred"]),
        "mean_car": None if primary is None else _num(primary["mean"]),
        "vif_max": vif_max,
        "placebos": placebos,
        "packages": {name: _pkg(name) for name in ("numpy", "pandas", "scipy", "statsmodels", "pyfixest", "pyarrow", "matplotlib", "streamlit", "plotly", "akshare")},
        "outputs": dict(result.outputs or {}),
        "note": (
            "复现需要同一配置、同一随机种子、同一事件表和同一行情。"
            "events_hash 只覆盖进入估计的事件，不包含被剔除的行。"
            "网站每次运行写在独立目录里，压缩包才是这次结果的副本。"
            "市场模型异常收益不是平均处理效应。"
        ),
    }
