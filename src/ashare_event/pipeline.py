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

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ashare_event.config import StudyConfig, clone_config, load_config, window_suffix
from ashare_event.diagnostics import variable_dictionary, variance_inflation
from ashare_event.figures import save_car_path, save_placebo, write_captions
from ashare_event.grouping import build_group_frames, write_group_figures
from ashare_event.inference import RBAR_MAX_EVENTS, bh_adjust, mean_residual_correlation, summarize_window
from ashare_event.manifest import build_manifest
from ashare_event.market import (
    load_or_download_index,
    load_or_download_stock,
    read_price_file,
    resolve_raw_dir,
    sample_date_range,
)
from ashare_event.methods_text import render_methods
from ashare_event.placebo import run_placebos
from ashare_event.regression import fit_car_regressions, regression_covariates
from ashare_event.returns import add_peer_count, estimate_events, winsorize_events
from ashare_event.schema import read_events_csv
from ashare_event.simulate import simulate
from ashare_event.tables import (
    DROP_LABELS,
    DROP_REASON_ORDER,
    attrition_table,
    car_table,
    diagnostics_table,
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
    regression_notes: str = ""
    methods_zh: str = ""
    methods_en: str = ""
    manifest: dict = field(default_factory=dict)
    groups: dict = field(default_factory=dict)
    vif: pd.DataFrame | None = None
    dictionary: pd.DataFrame | None = None
    anticipation: dict | None = None
    dropped_events: pd.DataFrame | None = None
    attrition: pd.DataFrame | None = None


def _load_inputs(
    cfg: StudyConfig,
    rng: np.random.Generator,
    industries,
    event_types,
    events_frame: pd.DataFrame | None = None,
    prices_frame: pd.DataFrame | None = None,
    market_frame: pd.DataFrame | None = None,
):
    """Return kept events, prices, market, simulated flag, the pre-filter count, and screened rows."""
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
    events, screened = _filter(events, industries, event_types)
    if events.empty:
        raise ValueError("筛选之后没有事件。放宽行业或事件类型。")
    if cfg.mode == "akshare":
        start, end = sample_date_range(events, cfg.estimation_window[0], cfg.placebo_offsets[0])
        raw = resolve_raw_dir(cfg.root)
        codes = sorted(events["code"].unique())
        workers = max(1, int(os.environ.get("ASHARE_DOWNLOAD_WORKERS", "4")))

        def _one(code: str):
            return load_or_download_stock(code, start, end, cfg.adjust, raw)

        if workers == 1 or len(codes) <= 1:
            frames = [_one(code) for code in codes]
        else:
            with ThreadPoolExecutor(max_workers=min(workers, len(codes))) as pool:
                frames = list(pool.map(_one, codes))
        prices = pd.concat(frames, ignore_index=True)
        market = load_or_download_index(cfg.benchmark, start, end, raw)
    return events, prices, market, simulated, loaded_n, screened


def _events_from_request(cfg: StudyConfig, events_frame: pd.DataFrame | None) -> pd.DataFrame:
    if events_frame is not None:
        if "event_id" in events_frame.columns and "code" in events_frame.columns:
            return events_frame.copy()
        from ashare_event.schema import load_events

        return load_events(events_frame)
    if not cfg.events_file:
        raise FileNotFoundError("需要事件表：在左侧上传 CSV，或在配置里填写 events_file。")
    return read_events_csv(cfg.resolve(cfg.events_file))


def _filter(events: pd.DataFrame, industries, event_types) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = pd.Series(True, index=events.index)
    if industries is not None:
        mask = mask & events["industry"].isin(list(industries))
    if event_types is not None:
        mask = mask & events["event_type"].isin(list(event_types))
    return events.loc[mask].reset_index(drop=True), events.loc[~mask].reset_index(drop=True)


def _sample_account(loaded: int, screened: pd.DataFrame, estimation, language: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Row-level drops and the count table. Every row except the first sums to ``loaded``."""
    labels = DROP_LABELS.get(language, DROP_LABELS["zh"])
    detail = ["event_id", "code", "event_date", "firm_name", "reason", "failed_window"]
    pieces: list[pd.DataFrame] = []
    if screened is not None and len(screened):
        rows = []
        for event in screened.itertuples(index=False):
            name = getattr(event, "firm_name", "")
            if name is None or (isinstance(name, float) and np.isnan(name)):
                name = ""
            stamp = pd.to_datetime(getattr(event, "event_date", None), errors="coerce")
            rows.append(
                {
                    "event_id": getattr(event, "event_id", ""),
                    "code": str(getattr(event, "code", "")).zfill(6)[-6:],
                    "event_date": "" if pd.isna(stamp) else pd.Timestamp(stamp).strftime("%Y-%m-%d"),
                    "firm_name": str(name),
                    "reason": "screened",
                    "failed_window": "",
                }
            )
        pieces.append(pd.DataFrame(rows, columns=detail))
    estimated_drop = estimation.dropped
    if estimated_drop is not None and len(estimated_drop):
        pieces.append(estimated_drop.reindex(columns=detail))
    dropped = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=detail)
    dropped["reason_zh"] = dropped["reason"].map(labels).fillna(dropped["reason"])
    counts = dropped["reason"].value_counts().to_dict() if len(dropped) else {}
    account = [{"step": labels["loaded"], "n": int(loaded), "reason": "loaded"}]
    seen: set[str] = set()
    for reason in (*DROP_REASON_ORDER, *(str(key) for key in counts)):
        if reason in seen:
            continue
        seen.add(reason)
        count = int(counts.get(reason, 0))
        if count:
            account.append({"step": labels.get(reason, reason), "n": count, "reason": reason})
    account.append({"step": labels["kept"], "n": int(len(estimation.events)), "reason": "kept"})
    attrition = pd.DataFrame(account, columns=["step", "n", "reason"])
    part = int(attrition.loc[attrition["reason"] != "loaded", "n"].sum())
    if part != int(loaded):
        raise ValueError(f"样本人数对不上：读入 {loaded}，分项合计 {part}。")
    return dropped, attrition


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
        "- 筛选和剔除的分项人数相加等于读入人数。明细在 dropped_events.csv，不改变谁进入 CAR。",
        f"- 公司数：{kept['firm_id'].nunique() if len(kept) else 0}",
        f"- 剔除：{reasons or '无'}",
        f"- 模型：{cfg.model}",
        f"- 估计窗：{list(cfg.estimation_window)}",
        f"- 回归窗：{list(cfg.regression_window)}",
        f"- 聚类：{cfg.cluster}",
        f"- 双向聚类：{'是' if cfg.two_way_cluster else '否'}",
        f"- 预事件窗口：{list(cfg.anticipation_window) if cfg.anticipation_window else '不计算'}",
        f"- 缩尾：{cfg.winsorize or 0}",
        f"- 市场领先滞后：{int(cfg.market_lags)}",
        "",
        "市场模型异常收益不是平均处理效应。横截面 t 不是首选检验。",
        "多个窗口和分组上的星号没有做多重检验校正。p BH 只调整窗口表里的首选 p 值。",
        "日历时间 t、广义符号 z 和 KP 调整后的秩检验是补充列，不改变首选规则。BHAR 与 CAR 分列存放。",
        "",
    ]
    if "limit_hit" in kept.columns and len(kept):
        n_hit = int(pd.Series(kept["limit_hit"]).fillna(False).astype(bool).sum())
        lines.append(
            f"- 事件日涨跌幅靠近涨跌停带宽：{n_hit}。"
            "这是按代码板块和公司名称做的启发式标记，不是交易所涨跌停状态表，程序没有因此删行。"
        )
    if "limit_hit_days" in kept.columns and len(kept):
        lines.append(
            f"- 事件窗内涨跌停天数合计：{int(pd.to_numeric(kept['limit_hit_days'], errors='coerce').fillna(0).sum())}。"
            "事件日当天的 limit_hit 仍单独保留。"
        )
    if "rolled_forward" in kept.columns and len(kept):
        lines.append(f"- 事件日向后对齐：{int(kept['rolled_forward'].astype(bool).sum())}。原因可以是周末、节假日，或盘后公告。")
    if len(kept) > RBAR_MAX_EVENTS:
        lines.append(f"- 平均残差相关在 {RBAR_MAX_EVENTS} 个事件的固定子样本上计算，只用来决定要不要改用 KP。")
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
    industry_frame: pd.DataFrame | None = None,
    factors_frame: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    on_step=None,
) -> StudyResult:
    from ashare_event.config import validate_windows

    cfg = clone_config(cfg)
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
    filtered, prices, market, simulated, loaded_n, screened = _load_inputs(
        cfg,
        rng,
        industries,
        event_types,
        events_frame=events_frame,
        prices_frame=prices_frame,
        market_frame=market_frame,
    )
    _step("estimate")
    compute_windows = list(cfg.event_windows)
    if cfg.anticipation_window is not None and tuple(cfg.anticipation_window) not in compute_windows:
        compute_windows.append(tuple(cfg.anticipation_window))
    estimation = estimate_events(
        filtered,
        prices,
        market,
        estimation_window=cfg.estimation_window,
        event_windows=compute_windows,
        path_window=cfg.path_window,
        regression_window=cfg.regression_window,
        model=cfg.model,
        min_estimation_obs=cfg.min_estimation_obs,
        require_complete_window=cfg.require_complete_window,
        industry_frame=industry_frame,
        factors_frame=factors_frame,
        market_lags=int(cfg.market_lags or 0),
    )
    if estimation.events.empty:
        counts = estimation.dropped["reason"].value_counts().to_dict()
        raise ValueError(f"没有事件通过估计窗和事件窗。剔除原因：{counts}")
    dropped_events, attrition = _sample_account(loaded_n, screened, estimation, cfg.language)
    events = add_peer_count(estimation.events)
    events["year"] = pd.to_datetime(events["event_date"]).dt.year.astype(int)
    suffix = window_suffix(*cfg.regression_window)
    raw_events = events
    if float(cfg.winsorize or 0) > 0:
        events = winsorize_events(events, float(cfg.winsorize))
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
    anticipation = None
    if cfg.anticipation_window is not None:
        anticipation = summarize_window(
            events,
            estimation.panel,
            tuple(cfg.anticipation_window),
            window_suffix(*cfg.anticipation_window),
            estimation_window=cfg.estimation_window,
            model=cfg.model,
            r_bar=r_bar,
            min_share=cfg.portfolio_min_share,
        )
    if not tests.empty:
        tests["p_bh"] = bh_adjust(tests["p_preferred"].to_numpy(dtype=float))
    path = _path_table(estimation.panel)
    covariates = regression_covariates(events, cfg.covariates, cfg.include_peer_count)
    regression_notes: list[str] = []
    models = fit_car_regressions(
        events,
        regression_window=cfg.regression_window,
        covariates=covariates,
        cluster=cfg.cluster,
        return_scale=cfg.return_scale,
        notes=regression_notes,
        two_way=cfg.two_way_cluster,
        wild_draws=cfg.wild_draws,
        rng=np.random.default_rng(cfg.seed + 17),
    )
    vif = variance_inflation(events, covariates)
    dictionary = variable_dictionary(events, [f"car_{suffix}", *covariates], cfg.language)
    if not vif.empty and float(pd.to_numeric(vif["vif"], errors="coerce").max() or 0) > 10:
        regression_notes.append("有变量的 VIF 超过 10，这些系数会不稳。VIF 不是删变量的因果标准。")
    actual = float(raw_events[f"car_{suffix}"].mean())
    placebos = {}
    if run_placebo and cfg.placebo_draws > 0:
        _step("placebo")
        placebos = run_placebos(
            events,
            prices,
            market,
            cfg,
            actual,
            rng,
            industry_frame=industry_frame,
            factors_frame=factors_frame,
        )
    groups = build_group_frames(events, estimation.panel, f"car_{suffix}", cfg.return_scale, cfg.language)
    theory = render_theory(
        sorted(events["event_type"].dropna().unique().tolist()),
        list(events.columns),
        cfg.language,
    )
    log = _sample_log(cfg, loaded_n, len(filtered), estimation, tests, simulated)
    if cfg.anticipation_window is not None:
        left, right = cfg.anticipation_window
        overlap = any(not (right < window[0] or window[1] < left) for window in cfg.event_windows)
        if overlap:
            log += "\n预事件窗口和某个报告窗有重叠。预事件行仍单独列出，不进入 BH，也不因此剔除事件。\n"
        else:
            log += "\n预事件窗口单独计算，不进入窗口表的 BH，也不因此剔除事件。\n"
    if int(cfg.market_lags or 0):
        log += "\n估计窗加入市场收益的前一日和后一日，斜率相加。事件窗预测只用同期市场收益。方差仍按单指数公式。\n"
    if float(cfg.winsorize or 0) > 0:
        log += "\n横截面 CAR 和 BHAR 做了缩尾。安慰剂对照的是缩尾前的平均 CAR。路径没有缩尾。\n"
    if regression_notes:
        log += "\n## 回归样本\n\n" + "\n".join(f"- {line}" for line in regression_notes) + "\n"
    for pack in placebos.values():
        if pack.get("sentence"):
            log += f"\n{pack['sentence']}\n"
    primary_row = tests[(tests["t1"] == cfg.regression_window[0]) & (tests["t2"] == cfg.regression_window[1])]
    preferred = str(primary_row.iloc[0]["preferred"]) if not primary_row.empty else ""
    methods_zh, methods_en = render_methods(
        cfg,
        n_events=len(events),
        n_firms=int(events["firm_id"].nunique()) if len(events) else 0,
        preferred=preferred,
        r_bar=float(r_bar),
        simulated=simulated,
    )
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
        regression_notes="\n".join(regression_notes),
        methods_zh=methods_zh,
        methods_en=methods_en,
        groups=groups,
        vif=vif,
        dictionary=dictionary,
        anticipation=anticipation,
        dropped_events=dropped_events,
        attrition=attrition,
    )
    if export:
        _step("export")
        result.outputs = export_result(result, dest=output_dir)
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


def export_result(result: StudyResult, dest: str | Path | None = None) -> dict[str, str]:
    cfg = result.config
    if dest is not None:
        root = Path(dest)
        paths = {
            "tables": root / "tables",
            "figures": root / "figures",
            "processed": root / "processed",
            "logs": root / "logs",
            "theory": root / "theory",
        }
    else:
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
    write_text(
        paths["tables"] / "table_diagnostics.tex",
        diagnostics_table(
            result.tests,
            result.events,
            getattr(result, "anticipation", None),
            cfg.language,
            cfg.return_scale,
            result.simulated,
        ),
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
    write_group_figures(result.groups or {}, paths["figures"], tuple(cfg.figure_size))
    write_text(paths["theory"] / "candidate_mechanisms.md", result.theory_md)
    if result.methods_zh:
        write_text(paths["theory"] / "methods_zh.md", result.methods_zh)
    if result.methods_en:
        write_text(paths["theory"] / "methods_en.md", result.methods_en)
    if result.attrition is not None and len(result.attrition):
        write_text(
            paths["tables"] / "table_attrition.tex",
            attrition_table(result.attrition, cfg.language, result.simulated),
        )
    if result.dropped_events is not None:
        result.dropped_events.to_csv(paths["processed"] / "dropped_events.csv", index=False)
    write_text(paths["logs"] / "sample_log.md", result.sample_log)
    frame = result.events.copy()
    frame["simulated"] = result.simulated
    frame.to_parquet(paths["processed"] / "event_car.parquet", index=False)
    frame.to_csv(paths["processed"] / "event_car.csv", index=False)
    result.tests.to_csv(paths["processed"] / "car_tests.csv", index=False)
    if result.vif is not None and len(result.vif):
        result.vif.to_csv(paths["processed"] / "vif.csv", index=False)
    if result.dictionary is not None and len(result.dictionary):
        result.dictionary.to_csv(paths["processed"] / "variable_dictionary.csv", index=False)
    for mode, pack in (result.placebos or {}).items():
        nulls = np.asarray(pack.get("nulls", []), dtype=float)
        if nulls.size:
            pd.DataFrame({"null_mean_car": nulls}).to_csv(paths["processed"] / f"placebo_{mode}.csv", index=False)
    if result.simulated:
        write_text(
            paths["processed"] / "SIMULATED.txt",
            "SIMULATED DATA\nevent_car 来自模拟路径。不能作为实证证据，不能写入论文。\n",
        )
    from ashare_event.replication import write_replication

    write_replication(result, paths["logs"] / "replication")
    result.outputs = {key: str(value) for key, value in paths.items()}
    result.manifest = build_manifest(result)
    write_text(paths["logs"] / "manifest.json", json.dumps(result.manifest, ensure_ascii=False, indent=2))
    return result.outputs


def run_from_file(config_path: str | Path, **kwargs) -> StudyResult:
    cfg = load_config(config_path)
    cfg.verbose = kwargs.pop("verbose", True)
    return run_study(cfg, **kwargs)
