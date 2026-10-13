"""Run one event study from a config file.

DECISIONS LOG (review before any number enters a paper)
=======================================================
Estimand   : mean cumulative abnormal return relative to a market model.
             Not an average treatment effect.
Unit       : firm-event for CAR; firm-event-day for the path.
Treatment  : the event date on the user's list. No control group.
Outcome    : simple return minus the pre-event market-model prediction.
Cluster    : firm_id, unless the config says otherwise.
Inference  : BMP when dates are dispersed; Kolari-Pynnonen when residuals
             are correlated; portfolio time-series test when event dates
             coincide. The cross-sectional t is reported and is not preferred.
Sample     : filters and drops are counted in the sample log.
Reproducibility: seed in config/study.yml; simulated draws use that seed.

ASSUMPTIONS [HUMAN: please confirm]:
- 前复权 closes when mode is akshare.
- A non-trading event date rolls forward by at most 10 calendar days.
- A missing day inside a CAR window drops that CAR.
- firm_id is stable and is the right cluster.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ashare_event.config import StudyConfig, load_config, window_suffix
from ashare_event.figures import save_car_path, save_placebo, write_captions
from ashare_event.inference import mean_residual_correlation, summarize_window
from ashare_event.market import (
    load_or_download_index,
    load_or_download_stock,
    read_price_file,
    sample_date_range,
)
from ashare_event.placebo import run_placebos
from ashare_event.regression import fit_car_regressions, regression_covariates
from ashare_event.returns import add_peer_count, estimate_events
from ashare_event.schema import read_events_csv
from ashare_event.simulate import simulate
from ashare_event.tables import (
    car_table,
    placebo_table,
    regression_table,
    summary_table,
    write_text,
)
from ashare_event.theory import render_theory


@dataclass
class StudyResult:
    config: StudyConfig
    events: pd.DataFrame
    panel: pd.DataFrame
    tests: pd.DataFrame
    path: pd.DataFrame
    placebos: dict
    models: list
    theory_md: str
    sample_log: str
    simulated: bool
    outputs: dict = field(default_factory=dict)
    r_bar: float = 0.0


def _load_inputs(
    cfg: StudyConfig,
    rng: np.random.Generator,
    industries,
    event_types,
    events_frame: pd.DataFrame | None = None,
    prices_frame: pd.DataFrame | None = None,
    market_frame: pd.DataFrame | None = None,
):
    """Return filtered events, prices, market, simulated flag, and the pre-filter count."""
    if cfg.mode == "demo":
        events, prices, market = simulate(cfg, rng)
        simulated = True
    elif cfg.mode == "local":
        events = _events_from_request(cfg, events_frame)
        if prices_frame is None and not cfg.prices_file:
            raise FileNotFoundError("本地行情需要价格文件。")
        if market_frame is None and not cfg.market_file:
            raise FileNotFoundError("本地行情需要指数文件。")
        prices = prices_frame if prices_frame is not None else read_price_file(cfg.resolve(cfg.prices_file))
        market = market_frame if market_frame is not None else read_price_file(cfg.resolve(cfg.market_file))
        simulated = False
    elif cfg.mode == "akshare":
        events = _events_from_request(cfg, events_frame)
        if events["firm_name"].astype(str).str.contains("示例").any():
            raise ValueError("事件表里还有示例行。换成自己的公告日期和公司后再用 AKShare 下载。")
        simulated = False
        prices = None
        market = None
    else:
        raise ValueError(cfg.mode)
    loaded_n = len(events)
    events = _filter(events, industries, event_types)
    if events.empty:
        raise ValueError("筛选之后没有事件。放宽行业或事件类型。")
    if cfg.mode == "akshare":
        start, end = sample_date_range(events, cfg.estimation_window[0], cfg.placebo_offsets[0])
        raw = cfg.root / "data" / "raw"
        frames = [
            load_or_download_stock(code, start, end, cfg.adjust, raw)
            for code in sorted(events["code"].unique())
        ]
        prices = pd.concat(frames, ignore_index=True)
        market = load_or_download_index(cfg.benchmark, start, end, raw)
    return events, prices, market, simulated, loaded_n


def _events_from_request(cfg: StudyConfig, events_frame: pd.DataFrame | None) -> pd.DataFrame:
    if events_frame is not None:
        if "event_id" in events_frame.columns and "code" in events_frame.columns:
            return events_frame.copy()
        from ashare_event.schema import load_events

        return load_events(events_frame)
    if not cfg.events_file:
        raise FileNotFoundError("需要事件表：在左侧上传 CSV，或在配置里填写 events_file。")
    return read_events_csv(cfg.resolve(cfg.events_file))


def _filter(events: pd.DataFrame, industries, event_types) -> pd.DataFrame:
    out = events
    if industries is not None:
        out = out[out["industry"].isin(list(industries))]
    if event_types is not None:
        out = out[out["event_type"].isin(list(event_types))]
    return out.reset_index(drop=True)


def _path_table(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty or "car_path" not in panel.columns:
        return pd.DataFrame(columns=["rel_day", "mean", "sd", "n", "se"])
    sub = panel.dropna(subset=["car_path"])
    if sub.empty:
        return pd.DataFrame(columns=["rel_day", "mean", "sd", "n", "se"])
    grouped = sub.groupby("rel_day")["car_path"]
    table = grouped.agg(mean="mean", sd="std", n="count").reset_index()
    table["se"] = table["sd"] / np.sqrt(table["n"].clip(lower=1))
    return table


def _sample_log(cfg: StudyConfig, loaded: int, filtered: int, estimation, tests: pd.DataFrame, simulated: bool) -> str:
    reasons = estimation.dropped["reason"].value_counts().to_dict() if not estimation.dropped.empty else {}
    kept = estimation.events
    lines = [
        "# 样本记录",
        "",
        f"- 模式：{cfg.mode}",
        f"- 模拟数据：{'是' if simulated else '否'}",
        f"- 读入事件：{loaded}",
        f"- 筛选后事件：{filtered}",
        f"- 进入 CAR 的事件：{len(kept)}",
        f"- 公司数：{kept['firm_id'].nunique() if len(kept) else 0}",
        f"- 剔除：{reasons or '无'}",
        f"- 模型：{cfg.model}",
        f"- 估计窗：{list(cfg.estimation_window)}",
        f"- 回归窗：{list(cfg.regression_window)}",
        f"- 聚类：{cfg.cluster}",
        "",
        "市场模型异常收益不是平均处理效应。横截面 t 不是首选检验。",
        "",
    ]
    if simulated:
        lines.append("本文件中的数字来自模拟 DGP，不能写入论文。")
        lines.append("")
    if not tests.empty:
        primary = tests[(tests["t1"] == cfg.regression_window[0]) & (tests["t2"] == cfg.regression_window[1])]
        if not primary.empty:
            row = primary.iloc[0]
            lines.append(
                f"回归窗首选检验：{row['preferred']}；平均残差相关 r_bar = {row['r_bar']:.3f}；"
                f"事件日最高重合比例 = {row['modal_share']:.2f}。"
            )
            lines.append("")
    return "\n".join(lines)


def run_study(
    cfg: StudyConfig,
    *,
    industries: list[str] | None = None,
    event_types: list[str] | None = None,
    regression_window: tuple[int, int] | None = None,
    run_placebo: bool = True,
    export: bool = True,
    events_frame: pd.DataFrame | None = None,
    prices_frame: pd.DataFrame | None = None,
    market_frame: pd.DataFrame | None = None,
    on_step=None,
) -> StudyResult:
    from ashare_event.config import validate_windows

    if regression_window is not None:
        cfg.regression_window = (int(regression_window[0]), int(regression_window[1]))
        if cfg.regression_window not in cfg.event_windows:
            cfg.event_windows = [*cfg.event_windows, cfg.regression_window]
        validate_windows(cfg)
    rng = np.random.default_rng(cfg.seed)

    def _step(key: str) -> None:
        if on_step is not None:
            on_step(key)

    _step("inputs")
    filtered, prices, market, simulated, loaded_n = _load_inputs(
        cfg,
        rng,
        industries,
        event_types,
        events_frame=events_frame,
        prices_frame=prices_frame,
        market_frame=market_frame,
    )
    _step("estimate")
    estimation = estimate_events(
        filtered,
        prices,
        market,
        estimation_window=cfg.estimation_window,
        event_windows=cfg.event_windows,
        path_window=cfg.path_window,
        regression_window=cfg.regression_window,
        model=cfg.model,
        min_estimation_obs=cfg.min_estimation_obs,
        require_complete_window=cfg.require_complete_window,
    )
    if estimation.events.empty:
        counts = estimation.dropped["reason"].value_counts().to_dict()
        raise ValueError(f"没有事件通过估计窗和事件窗。剔除原因：{counts}")
    events = add_peer_count(estimation.events)
    events["year"] = pd.to_datetime(events["event_date"]).dt.year.astype(int)
    suffix = window_suffix(*cfg.regression_window)
    r_bar = mean_residual_correlation(estimation.panel, events["event_id"], cfg.rbar_min_periods)
    _step("tests")
    rows = []
    for window in cfg.event_windows:
        rows.append(
            summarize_window(
                events,
                estimation.panel,
                window,
                window_suffix(*window),
                estimation_window=cfg.estimation_window,
                model=cfg.model,
                r_bar=r_bar,
                min_share=cfg.portfolio_min_share,
            )
        )
    tests = pd.DataFrame(rows)
    path = _path_table(estimation.panel)
    covariates = regression_covariates(events, cfg.covariates, cfg.include_peer_count)
    models = fit_car_regressions(
        events,
        regression_window=cfg.regression_window,
        covariates=covariates,
        cluster=cfg.cluster,
        return_scale=cfg.return_scale,
    )
    actual = float(events[f"car_{suffix}"].mean())
    placebos = {}
    if run_placebo and cfg.placebo_draws > 0:
        _step("placebo")
        placebos = run_placebos(events, prices, market, cfg, actual, rng)
    theory = render_theory(
        sorted(events["event_type"].dropna().unique().tolist()),
        list(events.columns),
        cfg.language,
    )
    log = _sample_log(cfg, loaded_n, len(filtered), estimation, tests, simulated)
    result = StudyResult(
        config=cfg,
        events=events,
        panel=estimation.panel,
        tests=tests,
        path=path,
        placebos=placebos,
        models=models,
        theory_md=theory,
        sample_log=log,
        simulated=simulated,
        r_bar=r_bar,
    )
    if export:
        _step("export")
        result.outputs = export_result(result)
    if cfg.verbose:
        primary = tests[(tests.t1 == cfg.regression_window[0]) & (tests.t2 == cfg.regression_window[1])].iloc[0]
        scale = cfg.return_scale
        print(
            f"N={int(primary['n'])}  平均CAR={primary['mean'] * scale:.3f}  "
            f"首选={primary['preferred']}  t={primary['t_' + ('portfolio' if primary['preferred']=='portfolio' else primary['preferred'])]:.2f}",
            flush=True,
        )
        if simulated:
            print("这些数字来自模拟样本，不能写入论文。", flush=True)
    return result


def export_result(result: StudyResult) -> dict[str, str]:
    cfg = result.config
    paths = {
        "tables": cfg.resolve(cfg.paths.get("tables", "paper/tabs")),
        "figures": cfg.resolve(cfg.paths.get("figures", "paper/figs")),
        "processed": cfg.resolve(cfg.paths.get("processed", "data/processed")),
        "logs": cfg.resolve(cfg.paths.get("logs", "output/logs")),
        "theory": cfg.resolve(cfg.paths.get("theory", "output/theory")),
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    suffix = window_suffix(*cfg.regression_window)
    covariates = regression_covariates(result.events, cfg.covariates, cfg.include_peer_count)
    write_text(
        paths["tables"] / "table_car.tex",
        car_table(result.tests, cfg.language, cfg.return_scale, result.simulated, cfg.cluster),
    )
    write_text(
        paths["tables"] / "table_summary.tex",
        summary_table(result.events, f"car_{suffix}", covariates, cfg.language, cfg.return_scale, result.simulated),
    )
    write_text(
        paths["tables"] / "table_regression.tex",
        regression_table(result.models, cfg.language, cfg.return_scale, result.simulated),
    )
    write_text(
        paths["tables"] / "table_placebo.tex",
        placebo_table(result.placebos, cfg.language, cfg.return_scale, result.simulated),
    )
    captions = {
        "fig:car_path": save_car_path(
            result.path,
            paths["figures"] / "fig_car_path.pdf",
            language=cfg.language,
            scale=cfg.return_scale,
            r_bar=result.r_bar,
            size=tuple(cfg.figure_size),
        )
    }
    for mode, placebo in result.placebos.items():
        captions[f"fig:placebo_{mode}"] = save_placebo(
            placebo,
            paths["figures"] / f"fig_placebo_{mode}.pdf",
            language=cfg.language,
            scale=cfg.return_scale,
            size=tuple(cfg.figure_size),
        )
    write_captions(paths["figures"] / "captions.tex", captions)
    write_text(paths["theory"] / "candidate_mechanisms.md", result.theory_md)
    write_text(paths["logs"] / "sample_log.md", result.sample_log)
    frame = result.events.copy()
    frame["simulated"] = result.simulated
    frame.to_parquet(paths["processed"] / "event_car.parquet", index=False)
    frame.to_csv(paths["processed"] / "event_car.csv", index=False)
    result.tests.to_csv(paths["processed"] / "car_tests.csv", index=False)
    if result.simulated:
        write_text(
            paths["processed"] / "SIMULATED.txt",
            "SIMULATED DATA\nevent_car 来自模拟路径。不能作为实证证据，不能写入论文。\n",
        )
    return {key: str(value) for key, value in paths.items()}


def run_from_file(config_path: str | Path, **kwargs) -> StudyResult:
    cfg = load_config(config_path)
    cfg.verbose = kwargs.pop("verbose", True)
    return run_study(cfg, **kwargs)
