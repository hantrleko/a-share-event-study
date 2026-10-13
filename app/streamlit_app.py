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
from ashare_event.tables import stars as star_mark

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
EXAMPLE_EVENTS = {
    "ai_policy": [
        {"code": "002230", "event_date": "2023-07-13", "event_type": "ai_policy", "industry": "软件", "firm_name": "科大讯飞（示例）"},
        {"code": "688111", "event_date": "2023-07-13", "event_type": "ai_policy", "industry": "软件", "firm_name": "金山办公（示例）"},
        {"code": "300418", "event_date": "2023-07-13", "event_type": "ai_policy", "industry": "互联网", "firm_name": "昆仑万维（示例）"},
        {"code": "000977", "event_date": "2023-07-13", "event_type": "ai_policy", "industry": "计算机", "firm_name": "浪潮信息（示例）"},
    ],
    "esg_guide": [
        {"code": "600519", "event_date": "2024-04-12", "event_type": "esg_guideline", "industry": "食品饮料", "firm_name": "贵州茅台（示例）"},
        {"code": "601318", "event_date": "2024-04-12", "event_type": "esg_guideline", "industry": "非银金融", "firm_name": "中国平安（示例）"},
        {"code": "600900", "event_date": "2024-04-12", "event_type": "esg_guideline", "industry": "公用事业", "firm_name": "长江电力（示例）"},
        {"code": "601012", "event_date": "2024-04-12", "event_type": "esg_guideline", "industry": "电力设备", "firm_name": "隆基绿能（示例）"},
    ],
}

st.set_page_config(
    page_title="EF Lab | A股事件研究",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
LOGO = ROOT / "assets" / "eugene-finance-logo.jpg"


def _apply_style() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"] { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; }
        .stApp { background: #f4f7fb; color: #16324f; }
        header[data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stAppDeployButton"],
        .stAppDeployButton,
        .stDeployButton,
        #MainMenu { display: none !important; }
        .block-container { padding-top: 0.45rem; padding-bottom: 2.2rem; max-width: 1180px; }
        section[data-testid="stSidebar"] {
          background: #eef3f8;
          border-right: 1px solid #d5dee8;
        }
        section[data-testid="stSidebar"] > div { padding-top: 0.85rem; }
        section[data-testid="stSidebar"] h2 {
          font-size: 0.72rem;
          letter-spacing: 0.16em;
          color: #5a6a7a;
          font-weight: 600;
          margin-top: 0.4rem;
          padding-top: 0.85rem;
          border-top: 1px solid #d5dee8;
        }
        section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p { font-size: 0.86rem; }
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
          background: #ffffff;
          border-color: #d5dee8;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 0.15rem; border-bottom: 1px solid #d5dee8; }
        .stTabs [data-baseweb="tab"] { background: transparent; color: #5a6a7a; border-radius: 0; }
        .stTabs [aria-selected="true"] { color: #1f4e79; border-bottom: 2px solid #1f4e79; }
        div[data-testid="stAlert"] {
          background: #f4f7fb;
          border: 0;
          border-left: 3px solid #1f4e79;
          border-radius: 2px;
          color: #16324f;
        }
        div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
          border: 1px solid #d5dee8;
          border-radius: 2px;
          background: #ffffff;
        }
        div[data-testid="stImage"] {
          background: #ffffff;
          border: 1px solid #d5dee8;
          padding: 0.35rem;
        }
        .stButton > button[kind="primary"] {
          background: #1f4e79;
          border: 1px solid #1f4e79;
        }
        .brand h1 {
          margin: 0;
          color: #1f4e79;
          font-size: 1.65rem;
          font-weight: 650;
          letter-spacing: -0.02em;
          line-height: 1.25;
        }
        .brand .subtitle { margin: 0.35rem 0 0; color: #5a6a7a; font-size: 0.92rem; }
        .kpi-row { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.55rem; margin: 0.4rem 0 0.7rem; }
        .kpi {
          background: #ffffff;
          border: 1px solid #d5dee8;
          border-radius: 8px;
          padding: 0.62rem 0.75rem 0.55rem;
          box-shadow: 0 1px 2px rgba(22, 50, 79, 0.05), 0 6px 16px rgba(22, 50, 79, 0.04);
        }
        .kpi span { display: block; font-size: 0.72rem; letter-spacing: 0.04em; color: #5a6a7a; }
        .kpi strong {
          display: block; margin-top: 0.18rem; font-size: 1.28rem; font-weight: 560;
          font-variant-numeric: tabular-nums; color: #16324f;
        }
        .kpi.neg strong { color: #9b3a3a; }
        .kpi.pos strong { color: #1d6b45; }
        .site-footer {
          margin-top: 1.6rem; padding-top: 0.75rem; border-top: 1px solid #d5dee8;
          color: #5a6a7a; font-size: 0.82rem;
        }
        .lesson {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.7rem;
  margin: 0.2rem 0 0.4rem;
}
.lesson article {
  background: #ffffff;
  border: 1px solid #d5dee8;
  border-radius: 8px;
  padding: 0.85rem 0.95rem 0.8rem;
  box-shadow: 0 1px 2px rgba(22, 50, 79, 0.05);
}
.lesson strong { display: block; color: #1f4e79; margin-bottom: 0.35rem; font-size: 0.95rem; }
        .lesson p { margin: 0; color: #3d4c5c; font-size: 0.92rem; line-height: 1.6; }
        .window-row {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 0.55rem;
          margin: 0.2rem 0 0.7rem;
        }
        .window-row article {
          background: #ffffff;
          border: 1px solid #d5dee8;
          border-radius: 8px;
          padding: 0.7rem 0.8rem 0.65rem;
          box-shadow: 0 1px 2px rgba(22, 50, 79, 0.05);
        }
        .window-row strong { display: block; color: #5a6a7a; font-size: 0.78rem; font-weight: 600; }
        .window-row b {
          display: block; margin-top: 0.15rem; color: #16324f; font-size: 1.2rem; font-weight: 560;
          font-variant-numeric: tabular-nums;
        }
        .window-row span { display: block; margin-top: 0.2rem; color: #5a6a7a; font-size: 0.78rem; }
        @media (max-width: 900px) {
          .kpi-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          .brand h1 { font-size: 1.25rem; }
          .lesson { grid-template-columns: 1fr; }
          .window-row { grid-template-columns: 1fr; }
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


def _drop_close_events(frame: pd.DataFrame, window: tuple[int, int]) -> pd.DataFrame:
    """Keep the earlier event when the same firm has another within the CAR span."""
    span = max(1, int(window[1]) - int(window[0]))
    work = frame.copy()
    firm_col = "firm_id" if "firm_id" in work.columns else "code"
    work["_when"] = pd.to_datetime(work["event_date"])
    keep: list = []
    for _, part in work.sort_values("_when").groupby(firm_col, sort=False):
        last = None
        for idx, row in part.iterrows():
            if last is None or (row["_when"] - last).days > span:
                keep.append(idx)
                last = row["_when"]
    return work.loc[keep].drop(columns="_when")


def _has_example_name(frame: pd.DataFrame | None) -> bool:
    if frame is None or frame.empty or "firm_name" not in frame.columns:
        return False
    return bool(frame["firm_name"].astype(str).str.contains("示例").any())


def _stat_text(value) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "—"
    return f"{float(number):.2f}"


def _regression_preview(models: list[dict]) -> pd.DataFrame:
    from ashare_event.labels import coef_label

    if not models:
        return pd.DataFrame()
    order: list[str] = []
    for model in models:
        for term in model["terms"]:
            if term not in order and not str(term).startswith("C("):
                order.append(str(term))
    rows: list[dict] = []
    for term in order:
        coef_row = {"项": coef_label("zh", term)}
        se_row = {"项": ""}
        for model in models:
            title = model["title"]
            if term not in model["b"]:
                coef_row[title] = ""
                se_row[title] = ""
                continue
            coef_row[title] = f"{model['b'][term]:.3f}{star_mark(model['p'][term])}"
            se_row[title] = f"({model['se'][term]:.3f})"
        rows.extend([coef_row, se_row])
    extras = [
        ("观测数", [str(model["n"]) for model in models]),
        ("聚类数", ["—" if not model["g"] else str(model["g"]) for model in models]),
        ("行业固定效应", ["是" if model["fe_industry"] else "否" for model in models]),
        ("年份固定效应", ["是" if model["fe_year"] else "否" for model in models]),
    ]
    for label, values in extras:
        row = {"项": label}
        for model, value in zip(models, values, strict=True):
            row[model["title"]] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _overlap_counts(frame: pd.DataFrame, window: tuple[int, int]) -> tuple[int, int]:
    if frame is None or frame.empty:
        return 0, 0
    span = max(1, int(window[1]) - int(window[0]))
    firm_col = "firm_id" if "firm_id" in frame.columns else "code"
    sizes = frame.groupby(firm_col).size()
    multi = int((sizes > 1).sum())
    close_pairs = 0
    dated = frame.copy()
    dated["_when"] = pd.to_datetime(dated["event_date"])
    for _, part in dated.groupby(firm_col, sort=False):
        gaps = part["_when"].sort_values().diff().dt.days.dropna()
        close_pairs += int((gaps <= span).sum())
    return multi, close_pairs


def _load_named_example(key: str) -> None:
    frame = pd.DataFrame(EXAMPLE_EVENTS[key])
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    st.session_state.custom_vars = []
    st.session_state.event_grid = _align_grid(frame, [])
    st.session_state.var_error = ""
    _reset_editor()


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
    st.image(str(LOGO), width=148)
    st.caption("EF Lab · [eugenefinance.tech](https://eugenefinance.tech)")
    st.header("事件")
    source = st.radio("事件从哪来", ["演示样本", "我的事件表"], key="event_source")
    import_error = ""
    if source == "我的事件表":
        st.caption("标准列：code, event_date, event_type, industry, firm_name。示例：`002230,2023-07-13,ai_policy,软件,科大讯飞`。")
        if st.button("载入示例：人工智能管理办法 2023-07-13", key="ex_ai"):
            _load_named_example("ai_policy")
            st.rerun()
        if st.button("载入示例：可持续发展报告指引 2024-04-12", key="ex_esg"):
            _load_named_example("esg_guide")
            st.rerun()
        st.caption("公司名称里的「示例」表示这两日是政策发布日，不是各公司自己的公告。名称里还有「示例」时，不会用 AKShare 下载真实行情。核对日期和公司后，删掉「示例」再运行。")
        template_rows = [*EXAMPLE_EVENTS["ai_policy"], *EXAMPLE_EVENTS["esg_guide"]]
        template_csv = pd.DataFrame(template_rows).to_csv(index=False).encode("utf-8-sig")
        st.download_button("下载填写模板", template_csv, file_name="events_template.csv", mime="text/csv", key="dl-template")
        st.caption("CSV 第一行用标准列：code, event_date, event_type, industry, firm_name。日期写成 2023-07-13。后面可以再加你自己的数值列。中文表头也能识别，例如证券代码、公告日期、事件类型、行业。")
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

    st.header("样本")
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
        if source == "我的事件表":
            drop_overlap = st.checkbox(
                "同一公司事件相距不超过 CAR 窗时，只留较早一件",
                value=False,
                key="drop_overlap",
                help="默认不删。每条公司-事件等权进入平均 CAR，同一公司的多起事件在回归里按公司聚类。勾选后，用日历日近似去掉靠得太近的后一件。",
            )
        else:
            drop_overlap = False
    else:
        drop_overlap = False
        st.caption("先在标准表里写上代码和日期。")

    st.header("行情")
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
        st.caption("基准是这条市场指数，不是行业指数。行业差别看结果里的分组路径。若要用行业指数，把该指数的日期和收盘价当作本地基准上传。交易日按这条指数的日历对齐。非交易日的事件日向后顺延，最多 10 个日历日。默认前复权，避免除权日造出假的大跌。")
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

    st.header("窗口")
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
    st.header("模型")
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
    example_blocked = source == "我的事件表" and _has_example_name(event_ready)
    if source == "我的事件表" and event_ready is None:
        missing.append("事件表")
    if example_blocked:
        missing.append("示例行")
    if price_mode == "local" and (prices_frame is None or market_frame is None):
        missing.append("本地行情")
    if window_error or import_error:
        missing.append("窗口或事件表")
    run = st.button("运行并导出", type="primary", disabled=bool(missing))
    if source == "我的事件表" and event_ready is None:
        st.caption("还差：至少一行证券代码和事件日期。")
    if example_blocked:
        st.caption("公司名里还有「示例」，运行已停用。这是政策日示例，不是公司公告。核对后删掉「示例」再运行。")
    if source == "我的事件表" and st.button("清空事件表", key="clear_grid"):
        st.session_state.custom_vars = []
        st.session_state.event_grid = _blank_grid([])
        st.session_state.var_error = ""
        _reset_editor()
        st.rerun()

ready_count = 0 if event_ready is None else len(event_ready)
logo_col, title_col = st.columns([1, 5], vertical_alignment="center")
with logo_col:
    st.image(str(LOGO), width=112)
with title_col:
    st.markdown(
        """
<div class="brand">
  <p class="kicker">V0.3 · EUGENE FINANCE</p>
  <h1>EF Lab · A 股事件研究流水线</h1>
  <p class="subtitle">Interactive Event Study Pipeline for A-Shares · 仅供学习与研究，不构成投资建议</p>
</div>
        """,
        unsafe_allow_html=True,
    )
if source == "演示样本":
    st.badge("演示样本", color="orange")
elif example_blocked:
    st.badge(f"{ready_count} 个已载入事件 · 含示例，还不能运行", color="orange")
elif event_ready is None:
    st.badge("等待事件", color="gray")
else:
    st.badge(
        f"{ready_count} 个有效事件 · {'市场模型' if model == 'market' else '市场调整'} · CAR [{car_window[0]}, {car_window[1]}]",
        color="green",
    )

with st.expander("先看懂", expanded=False):
    st.markdown(
        """
<div class="lesson">
<article><strong>异常收益是什么</strong><p>先用事件前的行情看这只股票平常怎样跟着指数走。事件窗口里，实际涨跌扣掉这个平常走势，剩下的就是异常收益。把窗口里的异常收益加起来，是累计异常收益，也就是 CAR。</p></article>
<article><strong>为什么做安慰剂</strong><p>把事件日挪到别的交易日再算一遍平均 CAR。如果随便哪一天都能算出同样大的数，真实日期上的结果就没有那么特别。安慰剂不是对照组，也不能当成因果检验。</p></article>
<article><strong>平均 CAR 怎么读</strong><p>表上的 1 是 1 个百分点。星号来自首选检验：* 是 10%，** 是 5%，*** 是 1%。事件如果扎在同一天，横截面 t 会把显著性说高，这时看组合检验。这是统计偏离，不是投资建议。</p></article>
</div>
        """,
        unsafe_allow_html=True,
    )

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

if run or st.session_state.get("study_result") is not None:
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
    sent_events = event_ready
    if source != "演示样本" and drop_overlap and event_ready is not None:
        sent_events = _drop_close_events(event_ready, car_window)
    with st.status("正在计算累计异常收益…", expanded=True) as status:
        labels = {
            "inputs": "读取事件和行情",
            "estimate": "估计窗与事件窗",
            "tests": "窗口检验与路径",
            "placebo": "安慰剂",
            "export": "回归表与导出",
        }

        def _on_step(key: str) -> None:
            status.update(label=labels.get(key, key))

        try:
            result = run_study(
                cfg,
                industries=industries,
                event_types=event_types,
                regression_window=car_window,
                run_placebo=do_placebo and placebo_draws > 0,
                export=True,
                events_frame=None if source == "演示样本" else sent_events,
                prices_frame=prices_frame,
                market_frame=market_frame,
                on_step=_on_step,
            )
        except Exception as exc:
            status.update(label="没有跑完", state="error")
            st.error(str(exc))
            st.stop()
        status.update(label="计算完成", state="complete", expanded=False)
        st.session_state.study_result = result
        st.session_state.study_window = car_window
        st.session_state.study_source = source
  else:
    result = st.session_state.study_result
    cfg = result.config
    car_window = st.session_state.study_window
  if st.session_state.get("study_source") != source:
      st.warning("下面的数字还是上一次运行的结果。事件来源已经换了，再点一次「运行并导出」。")
  if result.simulated:
      st.warning("当前是模拟样本。下面的数字不能当作实证结果。")
  primary = result.tests[(result.tests.t1 == car_window[0]) & (result.tests.t2 == car_window[1])]
  if primary.empty:
      st.error("回归窗没有进入结果表。")
      st.stop()
  primary = primary.iloc[0]
  preferred_t = {"portfolio": primary["t_portfolio"], "kp": primary["t_kp"], "bmp": primary["t_bmp"]}[primary["preferred"]]
  car_value = float(primary["mean"]) * cfg.return_scale
  tone = "neg" if car_value < 0 else "pos"
  st.markdown(
      f"""
<div class="kpi-row">
<div class="kpi"><span>事件数</span><strong>{int(primary["n"])}</strong></div>
<div class="kpi {tone}"><span>平均 CAR（百分点）</span><strong>{car_value:.3f}</strong></div>
<div class="kpi"><span>首选检验</span><strong>{str(primary["preferred"]).upper()}</strong></div>
<div class="kpi {tone}"><span>首选 t</span><strong>{preferred_t:.2f}</strong></div>
</div>
      """,
      unsafe_allow_html=True,
  )
  preferred_note = {
      "bmp": "事件日比较分散，所以用 BMP。它允许事件窗的波动和估计窗不一样。",
      "kp": "公司之间的估计窗残差相关较高，所以用 KP。它在 BMP 的基础上把 t 缩小。",
      "portfolio": "很多事件落在同一天，所以用等权组合的时间序列检验。这一天是一次共同冲击。",
  }[primary["preferred"]]
  st.caption(f"窗口 [{car_window[0]}, {car_window[1]}]。星号来自首选检验。{preferred_note}")
  window_cards = []
  for row in result.tests.itertuples(index=False):
      mark = star_mark(row.p_preferred)
      chosen = {"portfolio": row.t_portfolio, "kp": row.t_kp, "bmp": row.t_bmp}[row.preferred]
      window_cards.append(
          "<article>"
          f"<strong>[{int(row.t1)}, {int(row.t2)}]</strong>"
          f"<b>{row.mean * cfg.return_scale:.3f}{mark}</b>"
          f"<span>N={int(row.n)} · 首选 t={_stat_text(chosen)}</span>"
          "</article>"
      )
  st.markdown(f'<div class="window-row">{"".join(window_cards)}</div>', unsafe_allow_html=True)

  from ashare_event.figures import save_group_bars, save_grouped_paths
  from ashare_event.inference import mean_t
  from ashare_event.config import window_suffix

  car_col = f"car_{window_suffix(*car_window)}"
  figure_dir = Path(result.outputs["figures"])

  def _grouped(column: str) -> pd.DataFrame:
      rows = []
      if result.events.empty or car_col not in result.events.columns or column not in result.events.columns:
          return pd.DataFrame(columns=["label", "mean", "se", "stars", "n"])
      for name, part in result.events.groupby(column):
          values = pd.to_numeric(part[car_col], errors="coerce").dropna().to_numpy(dtype=float)
          if values.size < 2:
              continue
          mean, _t, p_value = mean_t(values)
          se = float(values.std(ddof=1) / np.sqrt(values.size))
          rows.append({"label": str(name), "mean": mean * cfg.return_scale, "se": se * cfg.return_scale, "stars": star_mark(p_value), "n": int(values.size)})
      return pd.DataFrame(rows)

  industry_groups = _grouped("industry")
  type_groups = _grouped("event_type")
  if not industry_groups.empty:
      save_group_bars(industry_groups, figure_dir / "fig_car_by_industry.png", title="按行业的平均 CAR", size=tuple(cfg.figure_size))
  if not type_groups.empty:
      save_group_bars(type_groups, figure_dir / "fig_car_by_type.png", title="按事件类型的平均 CAR", size=tuple(cfg.figure_size))

  def _paths_for(column: str) -> pd.DataFrame:
      if result.panel.empty or "car_path" not in result.panel.columns or column not in result.events.columns:
          return pd.DataFrame(columns=["group", "rel_day", "mean", "se"])
      counts = result.events[column].value_counts()
      keep_names = set(counts.head(6).index)
      meta = result.events.loc[result.events[column].isin(keep_names), ["event_id", column]]
      merged = result.panel.dropna(subset=["car_path"]).merge(meta, on="event_id", how="inner")
      rows = []
      for (group, day), part in merged.groupby([column, "rel_day"]):
          values = pd.to_numeric(part["car_path"], errors="coerce").dropna().to_numpy(dtype=float)
          if values.size < 2:
              continue
          rows.append(
              {
                  "group": str(group),
                  "rel_day": int(day),
                  "mean": float(values.mean()) * cfg.return_scale,
                  "se": float(values.std(ddof=1) / np.sqrt(values.size)) * cfg.return_scale,
              }
          )
      return pd.DataFrame(rows)

  by_industry = _paths_for("industry")
  by_type = _paths_for("event_type")
  if by_industry["group"].nunique() > 1 if not by_industry.empty else False:
      save_grouped_paths(by_industry, figure_dir / "fig_car_path_by_industry.png", size=tuple(cfg.figure_size))
  if by_type["group"].nunique() > 1 if not by_type.empty else False:
      save_grouped_paths(by_type, figure_dir / "fig_car_path_by_type.png", size=tuple(cfg.figure_size))
  multi_firms, close_pairs = _overlap_counts(result.events, car_window)
  adjust_label = {"qfq": "前复权", "hfq": "后复权", "": "不复权"}.get(cfg.adjust, cfg.adjust)
  st.caption(
      f"有效事件 {len(result.events)}，公司 {result.events['firm_id'].nunique()}。"
      f"{multi_firms} 家公司有多起事件，{close_pairs} 对事件相距不超过当前 CAR 窗。"
      f"平均 CAR 对每条公司-事件等权；同一公司同一天不能重复；窗口重叠默认都保留，回归按公司聚类。"
      f"行情按 {cfg.benchmark} 的交易日对齐，价格为{adjust_label}。"
  )
  tab_fig, tab_table, tab_reg, tab_note, tab_file = st.tabs(["图", "窗口表", "回归", "机制备忘", "导出"])
  with tab_fig:
      view = st.radio(
          "图表",
          ["整体路径", "分组路径", "分组柱", "安慰剂"],
          horizontal=True,
          key="fig_view",
          help="路径按交易日累加。分组柱用的是侧栏里当前选中的 CAR 窗。",
      )
      path_fig = figure_dir / "fig_car_path.png"
      if view == "整体路径" and path_fig.exists():
          st.image(str(path_fig), caption="整体平均 CAR 路径。阴影是 95% 区间，竖线是事件日。")
      elif view == "分组路径":
          group_choice = st.radio("分组", ["行业", "事件类型"], horizontal=True, key="path_group")
          group_path = figure_dir / ("fig_car_path_by_industry.png" if group_choice == "行业" else "fig_car_path_by_type.png")
          if group_path.exists():
              st.image(str(group_path), caption="每条线是一组的平均 CAR 路径，阴影是该组的 1.96 倍标准误。组内不足 2 家的不画。最多 6 组。")
          else:
              st.caption("这一组没有足够的公司可以画多条路径。")
      elif view == "分组柱":
          industry_fig = figure_dir / "fig_car_by_industry.png"
          type_fig = figure_dir / "fig_car_by_type.png"
          bar_left, bar_right = st.columns(2)
          if industry_fig.exists():
              bar_left.image(str(industry_fig), caption="按行业。误差线是 95% 区间，星号来自该组的横截面 t。")
          if type_fig.exists():
              bar_right.image(str(type_fig), caption="按事件类型。星号：* 10%，** 5%，*** 1%。")
          if not industry_fig.exists() and not type_fig.exists():
              st.caption("分组里至少要有 2 家公司，才会画出柱图。")
      else:
          if not result.placebos:
              st.caption("左侧勾选「做安慰剂」后，这里出现安慰剂分布。")
          for mode_name in result.placebos:
              picture = figure_dir / f"fig_placebo_{mode_name}.png"
              if picture.exists():
                  st.image(str(picture), caption="安慰剂直方图。柱是挪走事件日后的平均 CAR，竖线是真实日期。")
  with tab_table:
      st.caption("每一行是一个累加窗口。平均 CAR 已换成百分点。显著性按首选检验标注。")
      show = result.tests.copy()
      show["平均 CAR"] = (show["mean"] * cfg.return_scale).round(3)
      show["显著性"] = show["p_preferred"].map(star_mark)
      for column, label in [("t_cs", "横截面 t"), ("t_bmp", "BMP t"), ("t_kp", "KP t"), ("t_portfolio", "组合 t")]:
          show[label] = show[column].map(_stat_text)
      show["窗口"] = show.apply(lambda row: f"[{int(row.t1)}, {int(row.t2)}]", axis=1)
      show["首选"] = show["preferred"].map({"bmp": "BMP", "kp": "KP", "portfolio": "组合"})
      st.dataframe(
          show[["窗口", "n", "平均 CAR", "显著性", "横截面 t", "BMP t", "KP t", "组合 t", "首选"]].rename(columns={"n": "N"}),
          hide_index=True,
          use_container_width=True,
      )
  with tab_reg:
      st.caption("系数已经换成百分点。括号里是标准误。星号按该系数自己的 p 值，没有做多重检验校正。这是相关，不是平均处理效应。")
      preview = _regression_preview(result.models)
      if preview.empty:
          st.caption("这次没有回归表。数值变量至少要有不同取值，而且通过估计窗的事件不能太少。")
      else:
          st.dataframe(preview, hide_index=True, use_container_width=True)
  with tab_note:
      st.caption("估计前的读法，用来对照你添加的变量。显著系数不能反过来证明机制已经成立。")
      st.markdown(result.theory_md)
  with tab_file:
      st.caption("CAR 表、回归表和 PNG 都来自同一次运行。演示样本和模拟试跑不能写进正文。")
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
                  key=f"dl-{name}",
              )
      png_names = [
          "fig_car_path.png",
          "fig_car_path_by_industry.png",
          "fig_car_path_by_type.png",
          "fig_car_by_industry.png",
          "fig_car_by_type.png",
          "fig_placebo_in_time.png",
          "fig_placebo_in_space.png",
      ]
      png_cols = st.columns(2)
      for index, name in enumerate(png_names):
          file_path = figure_dir / name
          if file_path.exists():
              png_cols[index % 2].download_button(
                  f"下载 {name}",
                  data=file_path.read_bytes(),
                  file_name=name,
                  mime="image/png",
                  use_container_width=True,
                  key=f"dl-{name}",
              )
      car_tex = saved / "table_car.tex"
      if car_tex.exists() and car_tex.read_text(encoding="utf-8").strip():
          with st.expander("CAR 表的 LaTeX 片段"):
              st.code(car_tex.read_text(encoding="utf-8"), language="latex")
      window_csv = result.tests.to_csv(index=False).encode("utf-8-sig")
      event_csv = result.events.to_csv(index=False).encode("utf-8-sig")
      file_cols[0].download_button("下载 CAR 窗口表 CSV", window_csv, "car_windows.csv", "text/csv", use_container_width=True, key="dl-windows-csv")
      file_cols[1].download_button("下载事件层 CAR CSV", event_csv, "event_car.csv", "text/csv", use_container_width=True, key="dl-event-csv")
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
        st.caption("运行后在图、窗口表、回归、机制备忘和导出之间切换。平均 CAR 的 1 是 1 个百分点。")

with st.expander("方法说明"):
    st.markdown(
        """
**市场模型。** 在估计窗里，为每家公司单独拟合「个股收益 = 截距 + 斜率 × 市场基准收益」。事件窗里的实际收益减去这条拟合值，才是异常收益。市场调整则不估计截距和斜率，直接减市场基准收益。基准默认是沪深 300，可以改成别的指数代码；这不是行业指数。行业差别放在分组图里看。

**估计窗。** 默认是事件日前第 250 个交易日到第 31 个交易日。它必须整段落在路径窗和所有 CAR 窗的左边。右端若碰到事件日，当天的大涨大跌会进入斜率，事件窗里的异常收益会被压向 0。那是前视，不是更干净的估计。

**怎么加总。** 每条公司-事件等权。同一公司同一天不能有两行。窗口彼此重叠的后续事件默认保留，回归按公司聚类。侧栏可以改成只留较早的一件，用日历日近似。

**横截面依赖。** 同一天的公告会让多家公司的异常收益一起动。若把它们当成独立样本，平均 CAR 的标准误会偏小。估计窗残差相关较高时，首选检验改用 KP，在 BMP 上把 t 缩小。一半以上事件落在同一天时，改用等权组合的时间序列检验，把这一天看成一次共同冲击。

**交易与复权。** 交易日按基准指数的日历对齐，停牌日算缺失。事件窗里缺一天，这个窗口的 CAR 整段不要。AKShare 默认用前复权收盘价计算简单收益率。不复权会在除权日造出假的大跌。非交易日的事件日向后顺延，最多 10 个日历日。这叫非同步：公告日落在周末时，价格反应记在下一个交易日。

**不要把星号读满。** 同时看 [-1, 1]、[-3, 3]、[-5, 5]，再按行业和事件类型切开，会提高「至少有一个显著」的机会。星号没有做多重检验校正。事件日扎在同一天时，横截面 t 会把显著性说高，这时看组合时间序列检验。回归系数是相关，不是平均处理效应。
        """
    )

st.markdown(
    '<p class="site-footer">Part of <a href="https://eugenefinance.tech">EF Lab · eugenefinance.tech</a> · V0.3 · 仅供学习与研究，不构成投资建议。行情来自 AKShare 或你上传的文件。</p>',
    unsafe_allow_html=True,
)
