"""Left-hand inputs for one event study.

Standard event columns stay fixed. Extra variables are named on the left
and become columns in the same table.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from ashare_event.config import ConfigError, load_config, validate_windows
from ashare_event.labels import event_type_label
from ashare_event.market import read_price_frame
from ashare_event.pipeline import run_study
from ashare_event.schema import (
    FIELD_LABELS,
    STANDARD_FIELDS,
    SchemaError,
    load_events,
    rename_to_standard,
)
from ashare_event.simulate import simulate

PRESET_VARIABLES = (
    ("signal_cost", "信号成本"),
    ("size_log", "对数规模"),
    ("legitimacy_gap", "合法性缺口"),
)
VARIABLE_HELP = {
    "signal_cost": "信号成本。披露或行动越难模仿，这个数越高。若市场把它当成可信信号，它越高，累计异常收益越高。",
    "size_log": "公司规模的对数。通常只作控制变量，本身不是一条机制。",
    "legitimacy_gap": "合法性缺口。事件前的信任或争议越大，这个数越高。负面事件上，它越高，累计异常收益越低。",
}
FIELD_HELP = {
    "code": "6位A股代码。行情按这个代码下载。",
    "event_date": "你定义的事件日，通常是公告日。非交易日会顺延到其后10个日历日内的第一个交易日。",
    "event_type": "你给这件事起的标签，只用于筛选。行情软件里没有这个分类。",
    "industry": "你填的行业。用来筛选，也可以作为回归里的行业固定效应。",
    "firm_name": "公司名称，方便辨认。名称里若还有「示例」，不会用它下载真实行情。",
}
NAME_OK = re.compile(r"^[A-Za-z\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]{0,40}$")

st.set_page_config(page_title="A股事件研究", layout="wide", initial_sidebar_state="expanded")


def _apply_style() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f4f1ea; }
        section[data-testid="stSidebar"] {
            background: #fbfaf6;
            border-right: 1px solid #e5e0d6;
        }
        section[data-testid="stSidebar"] h2 {
            font-size: 0.92rem;
            font-weight: 650;
            color: #1f4b6e;
            letter-spacing: 0.01em;
            margin-top: 0.4rem;
            padding-top: 0.85rem;
            border-top: 1px solid #e6e1d8;
        }
        section[data-testid="stSidebar"] h2:first-of-type {
            border-top: 0;
            padding-top: 0;
        }
        div[data-testid="stMetric"] {
            background: #fffdf9;
            border-radius: 14px;
        }
        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            font-weight: 600;
        }
        .stButton > button[kind="primary"] {
            background: #1f4b6e;
            border: 1px solid #1f4b6e;
            border-radius: 10px;
        }
        .stButton > button[kind="primary"]:hover {
            background: #163850;
            border-color: #163850;
        }
        div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
            border: 1px solid #e6e1d6;
            border-radius: 12px;
            overflow: hidden;
            background: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_apply_style()


@st.cache_data(show_spinner=False)
def _demo_events(config_path: str) -> pd.DataFrame:
    cfg = load_config(config_path)
    events, _, _ = simulate(cfg, np.random.default_rng(cfg.seed))
    return events


def _blank_grid(variables: list[dict], rows: int = 6) -> pd.DataFrame:
    data: dict[str, list] = {
        "code": [""] * rows,
        "event_date": [pd.NaT] * rows,
        "event_type": [""] * rows,
        "industry": [""] * rows,
        "firm_name": [""] * rows,
    }
    for variable in variables:
        data[variable["name"]] = [np.nan if variable["kind"] == "number" else ""] * rows
    return pd.DataFrame(data)


def _align_grid(frame: pd.DataFrame, variables: list[dict]) -> pd.DataFrame:
    aligned = frame.copy()
    for field, default in (
        ("code", ""),
        ("event_date", pd.NaT),
        ("event_type", ""),
        ("industry", ""),
        ("firm_name", ""),
    ):
        if field not in aligned.columns:
            aligned[field] = default
    for variable in variables:
        if variable["name"] not in aligned.columns:
            aligned[variable["name"]] = np.nan if variable["kind"] == "number" else ""
    order = [*STANDARD_FIELDS, *[variable["name"] for variable in variables]]
    return aligned.loc[:, order]


def _sync_editor_state() -> None:
    edited = st.session_state.get("event_editor")
    if isinstance(edited, pd.DataFrame):
        st.session_state.event_grid = edited


def _ensure_state() -> None:
    if "custom_vars" not in st.session_state:
        st.session_state.custom_vars = []
    if "event_grid" not in st.session_state:
        st.session_state.event_grid = _blank_grid(st.session_state.custom_vars)
    _sync_editor_state()
    st.session_state.event_grid = _align_grid(st.session_state.event_grid, st.session_state.custom_vars)


def _reset_editor() -> None:
    st.session_state.pop("event_editor", None)


def _add_variable(name: str, kind: str) -> None:
    clean = name.strip()
    existing = {variable["name"] for variable in st.session_state.custom_vars}
    if clean in STANDARD_FIELDS or clean in existing or clean in {"firm_id", "event_id"}:
        st.session_state.var_error = f"「{clean}」已经是标准列或已有变量。"
        return
    if not NAME_OK.fullmatch(clean):
        st.session_state.var_error = "变量名用中文或英文开头，不要空格和加减号。"
        return
    st.session_state.custom_vars.append({"name": clean, "kind": kind})
    st.session_state.event_grid = _align_grid(st.session_state.event_grid, st.session_state.custom_vars)
    st.session_state.var_error = ""
    _reset_editor()


def _remove_variable(name: str) -> None:
    st.session_state.custom_vars = [variable for variable in st.session_state.custom_vars if variable["name"] != name]
    frame = st.session_state.event_grid.drop(columns=[name], errors="ignore")
    st.session_state.event_grid = _align_grid(frame, st.session_state.custom_vars)
    _reset_editor()


def _import_csv(upload) -> None:
    raw = pd.read_csv(upload, dtype=str)
    renamed = rename_to_standard(raw)
    extras = []
    reserved = set(STANDARD_FIELDS) | {"firm_id", "event_id"}
    for column in renamed.columns:
        if column in reserved:
            continue
        numeric = pd.to_numeric(renamed[column], errors="coerce")
        kind = "number" if numeric.notna().mean() >= 0.5 else "text"
        extras.append({"name": str(column), "kind": kind})
    renamed["event_date"] = pd.to_datetime(renamed["event_date"], errors="coerce")
    st.session_state.custom_vars = extras
    st.session_state.event_grid = _align_grid(renamed, extras)
    _reset_editor()


def _usable_events(frame: pd.DataFrame) -> pd.DataFrame | None:
    work = frame.copy()
    work["code"] = work["code"].fillna("").astype(str).str.strip()
    dated = pd.to_datetime(work["event_date"], errors="coerce")
    work = work[work["code"].ne("") & work["code"].ne("nan") & dated.notna()]
    if work.empty:
        return None
    return load_events(work)


def _numeric_columns(frame: pd.DataFrame, variables: list[dict]) -> list[str]:
    names = [variable["name"] for variable in variables if variable["kind"] == "number"]
    found = []
    for name in names:
        if name not in frame.columns:
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        if int(values.nunique(dropna=True)) > 1:
            found.append(name)
    return found


def _window_pair(title: str, left_default: int, right_default: int, key: str, help_text: str) -> tuple[int, int]:
    left_col, right_col = st.columns(2)
    left = int(left_col.number_input(f"{title}起点", value=int(left_default), step=1, key=f"{key}_l", help=help_text))
    right = int(right_col.number_input(f"{title}终点", value=int(right_default), step=1, key=f"{key}_r", help=help_text))
    return left, right


def _levels(frame: pd.DataFrame, column: str) -> list[str]:
    if frame is None or column not in frame.columns:
        return []
    values = sorted(value for value in frame[column].dropna().astype(str).unique() if value and value != "unspecified")
    return values


def _column_config(variables: list[dict]) -> dict:
    config = {
        "code": st.column_config.TextColumn(FIELD_LABELS["code"], help=FIELD_HELP["code"], required=True),
        "event_date": st.column_config.DateColumn(FIELD_LABELS["event_date"], help=FIELD_HELP["event_date"], format="YYYY-MM-DD"),
        "event_type": st.column_config.TextColumn(FIELD_LABELS["event_type"], help=FIELD_HELP["event_type"]),
        "industry": st.column_config.TextColumn(FIELD_LABELS["industry"], help=FIELD_HELP["industry"]),
        "firm_name": st.column_config.TextColumn(FIELD_LABELS["firm_name"], help=FIELD_HELP["firm_name"]),
    }
    for variable in variables:
        help_text = VARIABLE_HELP.get(
            variable["name"],
            "你添加的变量。数值列会进入回归；文本列只用于辨认或筛选。",
        )
        if variable["kind"] == "number":
            config[variable["name"]] = st.column_config.NumberColumn(variable["name"], help=help_text)
        else:
            config[variable["name"]] = st.column_config.TextColumn(variable["name"], help=help_text)
    return config


def _show_guide() -> None:
    with st.expander("输入怎么填，输出怎么读", expanded=False):
        left, right = st.columns(2)
        left.markdown(
            "\n".join(
                [
                    "**输入**",
                    "",
                    "- **证券代码、事件日期**：确定「哪只股票、哪一天」。行情只围绕这些日期下载。",
                    "- **事件类型、行业**：你自己起的标签。用来挑选子集，也可以让行业进入固定效应。",
                    "- **自定义变量**：你认为能拉开累计异常收益的特征。数值列进回归；文本列不进回归。",
                    "- **估计窗**：事件日之前，用来拟合每家公司对指数的敏感度。必须整段停在事件窗左边。",
                    "- **CAR 窗**：把这段交易日的异常收益加总。0 是事件日，-1 是前一个交易日。回归和页面顶部的数字用这一窗。",
                    "- **路径窗**：图的横轴。每个点是从路径起点一直加到当天的平均累计异常收益。",
                    "- **市场模型**：先在估计窗里拟合「个股收益 = 截距 + 斜率 × 指数收益」，事件窗里减不掉的部分就是异常收益。",
                    "- **市场调整**：不估计截距和斜率，直接用个股收益减指数收益。",
                    "- **前复权**：分红送股之后价格仍能前后比较，算收益率时用这个。",
                ]
            )
        )
        right.markdown(
            "\n".join(
                [
                    "**输出**",
                    "",
                    "- **平均 CAR**：窗口内异常收益加总后，再对公司取平均。表上的 1 表示 1 个百分点。",
                    "- **首选检验**：事件日分散时用 BMP；公司残差彼此相关时用 KP；一半以上事件落在同一天时用组合时间序列检验。星号来自这一列。",
                    "- **横截面 t**：假设每家公司彼此独立。事件扎堆时它会把显著性说得过高，所以只作对照。",
                    "- **回归系数**：变量高一点时，累计异常收益相差多少个百分点。这是相关关系。",
                    "- **路径图**：竖线是事件日。阴影是平均累计异常收益的 95% 区间。",
                    "- **安慰剂图**：柱是把事件日挪走之后的平均累计异常收益，竖线是真实日期的结果。",
                    "- **候选机制**：估计前的三条读法，用来设计变量。它不是已经证实的结论。",
                    "- **tex 文件**：给论文直接引入。演示样本和模拟试跑的数字不能写入论文。",
                ]
            )
        )


config_path = str(ROOT / "config" / "study.yml")
base = load_config(config_path)
_ensure_state()

with st.sidebar:
    st.header("1. 事件表")
    source = st.radio("事件从哪来", ["演示样本", "我的事件表"], key="event_source")
    import_error = ""
    if source == "我的事件表":
        st.caption("标准列固定为证券代码、事件日期、事件类型、行业、公司名称。自定义变量会变成新列：数值列解释累计异常收益的差别，文本列只作标记。")
        for preset_name, preset_label in PRESET_VARIABLES:
            if preset_name not in {item["name"] for item in st.session_state.custom_vars}:
                if st.button(f"加常用变量：{preset_label}", key=f"preset-{preset_name}"):
                    _add_variable(preset_name, "number")
                    st.rerun()
        new_name = st.text_input("新变量名", key="new_var_name", placeholder="例如 signal_cost")
        st.caption("变量名会原样进入回归，用英文最稳妥。")
        new_kind = st.radio("变量类型", ["数值", "文本"], horizontal=True, key="new_var_kind")
        if st.button("加入变量", key="add_var"):
            _add_variable(new_name, "number" if new_kind == "数值" else "text")
            st.rerun()
        if st.session_state.get("var_error"):
            st.error(st.session_state.var_error)
        for variable in list(st.session_state.custom_vars):
            left, right = st.columns([3, 1])
            left.write(variable["name"])
            if right.button("移除", key=f"rm-{variable['name']}"):
                _remove_variable(variable["name"])
                st.rerun()
        upload = st.file_uploader("导入 CSV 到标准表", type=["csv"], key="events_csv")
        header = ",".join([*STANDARD_FIELDS, *[item["name"] for item in st.session_state.custom_vars]]) + "\n"
        st.download_button("下载当前表头", header, file_name="events.csv", mime="text/csv")
        if upload is not None:
            token = f"{upload.name}:{upload.size}"
            if st.session_state.get("imported_token") != token:
                try:
                    _import_csv(upload)
                    st.session_state.imported_token = token
                    st.rerun()
                except (SchemaError, ValueError) as exc:
                    import_error = str(exc)
                    st.session_state.imported_token = token
                    st.error(import_error)
    else:
        st.caption("演示样本只用来看流程，数字不能写入论文。")

    if source == "演示样本":
        event_ready = _demo_events(config_path)
    else:
        event_ready = None
        try:
            event_ready = _usable_events(st.session_state.event_grid)
        except (SchemaError, ValueError) as exc:
            import_error = import_error or str(exc)
            st.error(import_error)

    st.header("2. 纳入哪些行")
    industries: list[str] | None = None
    event_types: list[str] | None = None
    if event_ready is not None:
        type_levels = _levels(event_ready, "event_type")
        industry_levels = _levels(event_ready, "industry")
        token = "demo" if source == "演示样本" else "own"
        if type_levels:
            event_types = st.multiselect(
                "事件类型",
                type_levels,
                default=type_levels,
                format_func=lambda item: event_type_label(base.language, item),
                key=f"types-{token}",
                help="只保留打了这些标签的行。标签是你在事件表里写的，不是行情软件的分类。",
            )
        else:
            st.caption("事件类型还是空的，填进表里之后会在这里列出。")
        if industry_levels:
            industries = st.multiselect(
                "行业",
                industry_levels,
                default=industry_levels,
                key=f"ind-{token}",
                help="只保留这些行业。行业固定效应也只用留在样本里的行业。",
            )
        else:
            st.caption("行业还是空的。")
    else:
        st.caption("先在标准表里写上代码和日期。")

    st.header("3. 行情")
    if source == "演示样本":
        price_mode = "demo"
        st.caption("演示样本使用内置行情。")
    else:
        price_choice = st.radio("价格从哪来", ["AKShare", "本地文件"], key="price_mode")
        price_mode = "akshare" if price_choice == "AKShare" else "local"
    benchmark = base.benchmark
    adjust = base.adjust
    prices_frame = None
    market_frame = None
    if price_mode == "akshare":
        benchmark = st.text_input(
            "基准指数",
            value=base.benchmark,
            key="benchmark",
            help="市场收益用哪条指数。000300 是沪深300。个股异常收益是相对这条指数算的。",
        ).strip() or base.benchmark
        adjust_label = st.selectbox(
            "复权",
            ["前复权", "后复权", "不复权"],
            key="adjust",
            help="前复权让分红送股前后的价格能比较，算收益率时用它。不复权会在除权日出现假的大跌。",
        )
        adjust = {"前复权": "qfq", "后复权": "hfq", "不复权": ""}[adjust_label]
        st.caption("AKShare 只下载事件表里这些代码的日线，并写入 data/raw。已有文件不会覆盖。")
    elif price_mode == "local":
        price_file = st.file_uploader("个股行情 CSV（代码、日期、收盘）", type=["csv"], key="prices_csv")
        market_file = st.file_uploader("指数行情 CSV（日期、收盘）", type=["csv"], key="market_csv")
        if price_file is not None:
            try:
                prices_frame = read_price_frame(pd.read_csv(price_file))
            except Exception as exc:
                st.error(str(exc))
        if market_file is not None:
            try:
                market_frame = read_price_frame(pd.read_csv(market_file))
            except Exception as exc:
                st.error(str(exc))

    st.header("4. 窗口与回归")
    estimation = _window_pair(
        "估计窗",
        base.estimation_window[0],
        base.estimation_window[1],
        "est",
        "事件日之前的交易日，用来估计每家公司跟随指数的方式。右端必须早于路径窗和所有 CAR 窗。",
    )
    path_window = _window_pair(
        "路径窗",
        base.path_window[0],
        base.path_window[1],
        "path",
        "图上画出的区间。每个点是从左端累加到该交易日的平均累计异常收益。",
    )
    car_window = _window_pair(
        "CAR 窗",
        base.regression_window[0],
        base.regression_window[1],
        "car",
        "回归和页面顶部指标使用的累加区间。0 是事件日，负数是事件前的交易日。",
    )
    report_extra = []
    for preset in ((-1, 1), (-3, 3), (-5, 5)):
        if st.checkbox(
            f"同时报告 [{preset[0]}, {preset[1]}]",
            value=True,
            key=f"preset-w-{preset[0]}-{preset[1]}",
            help="额外给出这个窗口的平均累计异常收益，方便和主 CAR 窗对照。",
        ):
            report_extra.append(preset)
    event_windows = list(dict.fromkeys([*report_extra, car_window]))
    model = st.radio(
        "收益模型",
        ["market", "market_adjusted"],
        format_func=lambda item: "市场模型" if item == "market" else "市场调整",
        key="return_model",
        help="市场模型为每家公司估计自己的截距和斜率。市场调整则假定斜率是 1、截距是 0。",
    )
    covariate_source = event_ready if event_ready is not None else st.session_state.event_grid
    covariate_options = _numeric_columns(covariate_source, st.session_state.custom_vars) if source == "我的事件表" else []
    if source == "演示样本" and event_ready is not None:
        covariate_options = [column for column in ("signal_cost", "size_log", "legitimacy_gap") if column in event_ready.columns]
    covariates = []
    if covariate_options:
        covariates = st.multiselect(
            "放进回归的变量",
            covariate_options,
            default=covariate_options,
            key=f"cov-{source}",
            help="勾选的数值列用来解释公司之间 CAR 的差别。系数单位是百分点。这是相关，不是因果。",
        )
    elif source == "我的事件表":
        st.caption("数值变量要在表里有不同取值，才会出现在这里。")
    include_peer = st.checkbox(
        "加入同业事件数（内生，默认别当识别）",
        value=False,
        key="peer",
        help="同一行业、前后 5 个日历日内还有多少别的样本事件。它由样本自己数出来，不能单独当成因果识别。",
    )
    do_placebo = st.checkbox(
        "做安慰剂",
        value=False,
        key="placebo",
        help="把每家公司的事件日挪到估计窗之前的其他交易日，再重算平均 CAR。用来看真实日期是否突出。",
    )
    placebo_draws = int(
        st.number_input(
            "安慰剂次数",
            min_value=0,
            max_value=500,
            value=50,
            step=10,
            key="draws",
            help="挪动并重算的次数。次数越多，p 值越细，等待也越久。",
        )
    )

    window_error = None
    if estimation[0] > estimation[1] or path_window[0] > path_window[1] or car_window[0] > car_window[1]:
        window_error = "窗口起点要早于终点。"
        st.error(window_error)
    else:
        probe = load_config(config_path)
        probe.mode = price_mode
        probe.model = model
        probe.estimation_window = estimation
        probe.path_window = path_window
        probe.regression_window = car_window
        probe.event_windows = event_windows
        try:
            validate_windows(probe)
        except ConfigError as exc:
            window_error = str(exc)
            st.error(window_error)

    missing = []
    if source == "我的事件表" and event_ready is None:
        missing.append("事件表")
    if price_mode == "local" and (prices_frame is None or market_frame is None):
        missing.append("本地行情")
    if window_error or import_error:
        missing.append("窗口或事件表")
    run = st.button("运行并导出", type="primary", disabled=bool(missing))
    if source == "我的事件表" and event_ready is None:
        st.caption("还差：至少一行证券代码和事件日期。")
    if source == "我的事件表" and st.button("清空事件表", key="clear_grid"):
        st.session_state.custom_vars = []
        st.session_state.event_grid = _blank_grid([])
        st.session_state.var_error = ""
        _reset_editor()
        st.rerun()

ready_count = 0 if event_ready is None else len(event_ready)
head_left, head_right = st.columns([5, 2])
with head_left:
    st.markdown("## 事件研究")
    st.caption("左侧改事件、窗口和行情。这里看累计异常收益、图，以及可以放进论文的文件。")
with head_right:
    if source == "演示样本":
        st.badge("演示样本", color="orange")
        st.caption("数字不能写入论文")
    elif event_ready is None:
        st.badge("等待事件", color="gray")
        st.caption("写上代码和日期后再运行")
    else:
        st.badge(f"{ready_count} 个有效事件", color="green")
        st.caption(f"{'市场模型' if model == 'market' else '市场调整'} · CAR [{car_window[0]}, {car_window[1]}]")

if source == "演示样本":
    st.info("演示样本用来看流程。换成自己的研究时，在左侧切到「我的事件表」。")
else:
    st.subheader("标准事件表")
    st.caption("固定列在左，你添加的变量在右。把鼠标放在列名上可看含义。空白行不会进入估计。")
_show_guide()
if source != "演示样本":
    edited = st.data_editor(
        _align_grid(st.session_state.event_grid, st.session_state.custom_vars),
        num_rows="dynamic",
        column_config=_column_config(st.session_state.custom_vars),
        hide_index=True,
        key="event_editor",
    )
    st.session_state.event_grid = edited

if run:
    cfg = load_config(config_path)
    cfg.mode = price_mode
    cfg.model = model
    cfg.verbose = False
    cfg.benchmark = benchmark
    cfg.adjust = adjust
    cfg.estimation_window = estimation
    cfg.path_window = path_window
    cfg.regression_window = car_window
    cfg.event_windows = event_windows
    cfg.covariates = list(covariates)
    cfg.include_peer_count = include_peer
    cfg.placebo_draws = placebo_draws
    validate_windows(cfg)
    with st.spinner("正在估计。"):
        try:
            result = run_study(
                cfg,
                industries=industries,
                event_types=event_types,
                regression_window=car_window,
                run_placebo=do_placebo and placebo_draws > 0,
                export=True,
                events_frame=None if source == "演示样本" else event_ready,
                prices_frame=prices_frame,
                market_frame=market_frame,
            )
        except Exception as exc:
            st.error(str(exc))
            st.stop()
    if result.simulated:
        st.warning("当前是模拟样本。下面的数字不能当作实证结果。")
    primary = result.tests[(result.tests.t1 == car_window[0]) & (result.tests.t2 == car_window[1])]
    if primary.empty:
        st.error("回归窗没有进入结果表。")
        st.stop()
    primary = primary.iloc[0]
    metric_n, metric_mean, metric_pref, metric_t = st.columns(4)
    metric_n.metric("事件数", int(primary["n"]), border=True, help="通过估计窗、并且事件窗没有缺天的公司-事件。")
    metric_mean.metric("平均 CAR", f"{primary['mean'] * cfg.return_scale:.3f}", border=True, help="单位是百分点。1 表示 1 个百分点。")
    metric_pref.metric("首选检验", str(primary["preferred"]).upper(), border=True, help="星号来自这一检验，不来自横截面 t。")
    preferred_t = {"portfolio": primary["t_portfolio"], "kp": primary["t_kp"], "bmp": primary["t_bmp"]}[primary["preferred"]]
    metric_t.metric("首选 t", f"{preferred_t:.2f}", border=True)
    preferred_note = {
        "bmp": "事件日比较分散，所以用 BMP。它允许事件窗的波动和估计窗不一样。",
        "kp": "公司之间的估计窗残差相关较高，所以用 KP。它在 BMP 的基础上把 t 缩小。",
        "portfolio": "很多事件落在同一天，所以用等权组合的时间序列检验。这一天是一次共同冲击。",
    }[primary["preferred"]]
    st.caption(
        f"窗口 [{car_window[0]}, {car_window[1]}]。{preferred_note}"
    )
    tab_fig, tab_table, tab_note, tab_file = st.tabs(["图", "窗口表", "机制备忘", "导出"])
    with tab_fig:
        figure = Path(result.outputs["figures"]) / "fig_car_path.png"
        if figure.exists():
            st.image(str(figure), caption="横轴是相对事件日的交易日，纵轴是平均累计异常收益。竖线为事件日。")
        else:
            st.info("这次没有画出路径图。")
        if not result.placebos:
            st.caption("左侧勾选「做安慰剂」后，这里会多一张对照图。")
        for mode_name in result.placebos:
            picture = Path(result.outputs["figures"]) / f"fig_placebo_{mode_name}.png"
            if picture.exists():
                st.image(str(picture), caption="柱是挪走事件日之后的结果，竖线是真实日期。")
    with tab_table:
        st.caption("每一行是一个累加窗口。平均 CAR 已换成百分点。看「首选」列对应的 t。")
        show = result.tests.copy()
        show["平均 CAR"] = (show["mean"] * cfg.return_scale).round(3)
        for column, label in [("t_cs", "横截面 t"), ("t_bmp", "BMP t"), ("t_kp", "KP t"), ("t_portfolio", "组合 t")]:
            show[label] = show[column].round(2)
        show["窗口"] = show.apply(lambda row: f"[{int(row.t1)}, {int(row.t2)}]", axis=1)
        show["首选"] = show["preferred"].map({"bmp": "BMP", "kp": "KP", "portfolio": "组合"})
        st.dataframe(
            show[["窗口", "n", "平均 CAR", "横截面 t", "BMP t", "KP t", "组合 t", "首选"]].rename(columns={"n": "N"}),
            hide_index=True,
            use_container_width=True,
        )
    with tab_note:
        st.caption("估计前的读法，用来对照你添加的变量。显著系数不能反过来证明机制已经成立。")
        st.markdown(result.theory_md)
    with tab_file:
        st.caption("这些 tex 给论文直接引入。演示样本和模拟试跑不能写进正文。")
        file_cols = st.columns(2)
        saved = Path(result.outputs["tables"])
        names = ["table_car.tex", "table_summary.tex", "table_regression.tex", "table_placebo.tex"]
        for index, name in enumerate(names):
            file_path = saved / name
            if file_path.exists() and file_path.read_text(encoding="utf-8").strip():
                file_cols[index % 2].download_button(
                    f"下载 {name}",
                    data=file_path.read_text(encoding="utf-8"),
                    file_name=name,
                    mime="text/plain",
                    use_container_width=True,
                )
        with st.expander("样本怎么筛出来的"):
            st.markdown(result.sample_log)
else:
    step_a, step_b, step_c = st.columns(3)
    with step_a.container(border=True):
        st.markdown("**1. 放入事件**")
        st.caption("演示样本可以直接运行。自己的公告改到左侧「我的事件表」，在标准列里填写。")
    with step_b.container(border=True):
        st.markdown("**2. 加上变量**")
        st.caption("信号成本、规模、合法性缺口，或你自己起的数值变量。文本列只作标记。")
    with step_c.container(border=True):
        st.markdown("**3. 看结果**")
        st.caption("运行后在图、窗口表、机制备忘和导出之间切换。平均 CAR 的 1 是 1 个百分点。")
