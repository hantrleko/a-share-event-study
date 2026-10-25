"""Booktabs tables with threeparttable notes.

Numbers are written by this module from the estimation objects.
Do not copy them by hand into the manuscript.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ashare_event.config import window_label
from ashare_event.labels import coef_label, text


def stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def tex_escape(value) -> str:
    text_value = str(value)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ):
        text_value = text_value.replace(old, new)
    return text_value


def _num(value, digits=3) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def _wrap_table(caption: str, label: str, tabular: str, notes: list[str]) -> str:
    note_lines = "\n".join(f"\\item {tex_escape(note)}" for note in notes if note)
    return "\n".join(
        [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\begin{threeparttable}",
            f"\\caption{{{tex_escape(caption)}}}",
            f"\\label{{{label}}}",
            tabular,
            "\\begin{tablenotes}[flushleft]",
            "\\footnotesize",
            note_lines,
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{table}",
            "",
        ]
    )


def _notes(language: str, cluster: str, scale: float, simulated: bool, extra: list[str] | None = None) -> list[str]:
    notes = []
    if simulated:
        notes.append(text(language, "simulated"))
    notes.extend(
        [
            text(language, "not_att"),
            text(language, "stars"),
            text(language, "scale_note", scale=scale),
            text(language, "cluster_note", cluster=cluster),
            text(language, "preferred_note"),
        ]
    )
    if extra:
        notes.extend(extra)
    return notes


def car_table(tests: pd.DataFrame, language: str, scale: float, simulated: bool, cluster: str) -> str:
    header = " & ".join(
        [
            text(language, "window"),
            text(language, "n"),
            text(language, "mean"),
            text(language, "t_cs"),
            text(language, "t_bmp"),
            text(language, "t_kp"),
            text(language, "t_port"),
            text(language, "preferred"),
            "t 秩" if language == "zh" else "t rank",
            "p BH",
        ]
    )
    rows = [header + r" \\"]
    pref = {
        "portfolio": text(language, "pref_portfolio"),
        "kp": text(language, "pref_kp"),
        "bmp": text(language, "pref_bmp"),
    }
    for row in tests.itertuples(index=False):
        mean = row.mean * scale if np.isfinite(row.mean) else np.nan
        cells = [
            window_label(int(row.t1), int(row.t2)),
            str(int(row.n)),
            _num(mean) + stars(row.p_preferred),
            _num(row.t_cs, 2),
            _num(row.t_bmp, 2),
            _num(row.t_kp, 2),
            _num(row.t_portfolio, 2),
            pref.get(row.preferred, row.preferred),
            _num(getattr(row, "t_corrado", np.nan), 2),
            _num(getattr(row, "p_bh", np.nan), 3),
        ]
        rows.append(" & ".join(cells) + r" \\")
    tabular = "\n".join(
        [
            "\\begin{tabular}{lrrrrrrlrr}",
            "\\toprule",
            rows[0],
            "\\midrule",
            *rows[1:],
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    caption = "累计异常收益" if language == "zh" else "Cumulative abnormal returns"
    if language == "zh":
        extra = [
            "p BH 是对本表各窗口的首选 p 值做 Benjamini–Hochberg 调整。星号仍用未调整的首选 p 值。",
            "t 秩把估计窗残差和事件窗异常收益放在一起排名，用来对照厚尾。它不是另一套处理效应检验。",
        ]
    else:
        extra = [
            "p BH adjusts the preferred p-values in this table by Benjamini-Hochberg. Stars still use the unadjusted preferred p-value.",
            "The rank t pools estimation residuals and event-window abnormal returns. It is a thick-tail check, not a treatment-effect test.",
        ]
    return _wrap_table(caption, "tab:car", tabular, _notes(language, cluster, scale, simulated, extra))


def summary_table(events: pd.DataFrame, outcome: str, covariates: list[str], language: str, scale: float, simulated: bool) -> str:
    columns = [outcome, *[col for col in covariates if col in events.columns]]
    header = " & ".join(["", "N", "Mean" if language == "en" else "均值", "SD", "P25", "P50", "P75"])
    body = []
    for column in columns:
        values = pd.to_numeric(events[column], errors="coerce").dropna()
        if column == outcome:
            values = values * scale
            label = text(language, "mean")
        else:
            label = coef_label(language, column)
        if values.empty:
            continue
        stats = [
            tex_escape(label),
            str(int(values.size)),
            _num(values.mean()),
            _num(values.std(ddof=1)),
            _num(values.quantile(0.25)),
            _num(values.quantile(0.50)),
            _num(values.quantile(0.75)),
        ]
        body.append(" & ".join(stats) + r" \\")
    tabular = "\n".join(
        [
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            header + r" \\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    caption = "描述统计" if language == "zh" else "Summary statistics"
    notes = []
    if simulated:
        notes.append(text(language, "simulated"))
    notes.append(text(language, "not_att"))
    notes.append(text(language, "scale_note", scale=scale))
    return _wrap_table(caption, "tab:summary", tabular, notes)


def regression_table(models: list[dict], language: str, scale: float, simulated: bool) -> str:
    if not models:
        return ""
    order = []
    for model in models:
        for term in model["terms"]:
            if term not in order and not str(term).startswith("C("):
                order.append(term)
    header = " & ".join(["", *[model["title"] for model in models]]) + r" \\"
    body = []
    for term in order:
        coefs = []
        ses = []
        for model in models:
            if term not in model["b"]:
                coefs.append("")
                ses.append("")
                continue
            coefs.append(_num(model["b"][term]) + stars(model["p"][term]))
            ses.append(f"({_num(model['se'][term])})")
        body.append(" & ".join([tex_escape(coef_label(language, term)), *coefs]) + r" \\")
        body.append(" & ".join(["", *ses]) + r" \\")
    def yn(flag: bool) -> str:
        return text(language, "yes") if flag else text(language, "no")

    footer = []
    for key, label_key in (("fe_industry", "fe_industry"), ("fe_year", "fe_year")):
        footer.append(
            " & ".join([text(language, label_key), *[yn(model[key]) for model in models]]) + r" \\"
        )
    footer.append(" & ".join([text(language, "n"), *[str(model["n"]) for model in models]]) + r" \\")
    footer.append(
        " & ".join(
            [
                text(language, "clusters"),
                *[str(model["g"]) if model["g"] else "--" for model in models],
            ]
        )
        + r" \\"
    )
    footer.append(
        " & ".join(
            [text(language, "r2"), *[_num(model["r2"]) if not model["fe_industry"] and not model["fe_year"] else "--" for model in models]]
        )
        + r" \\"
    )
    footer.append(
        " & ".join(
            [text(language, "r2_within"), *[_num(model["r2_within"]) if model["r2_within"] is not None else "--" for model in models]]
        )
        + r" \\"
    )
    align = "l" + "c" * len(models)
    tabular = "\n".join(
        [
            f"\\begin{{tabular}}{{{align}}}",
            "\\toprule",
            header,
            "\\midrule",
            *body,
            "\\midrule",
            *footer,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    caption = "累计异常收益的横截面回归" if language == "zh" else "Cross-sectional regressions of CAR"
    cluster = models[0]["cluster"]
    extra = []
    if language == "zh":
        extra.append("固定效应规格报告组内 R²。没有固定效应的规格报告普通 R²。")
        extra.append("解释变量的系数是相关关系。事件研究本身不提供因果识别。")
    else:
        extra.append("Fixed-effect columns report within R². Columns without fixed effects report the ordinary R².")
        extra.append("Covariate coefficients are associations. The event study does not identify a causal effect.")
    if any(model.get("two_way") for model in models):
        extra.append(
            "标成双向的列按公司和事件日聚类。"
            if language == "zh"
            else "Columns marked two-way are clustered by firm and event date."
        )
    if any(model.get("wild_p") for model in models):
        extra.append(
            "不含固定效应的规格另有 wild cluster bootstrap p，写在运行快照里。星号仍来自聚类标准误。"
            if language == "zh"
            else "Specifications without fixed effects also have wild cluster bootstrap p-values in the run manifest. Stars still use the clustered standard errors."
        )
    return _wrap_table(caption, "tab:car_reg", tabular, _notes(language, cluster, scale, simulated, extra))


def placebo_table(placebos: dict, language: str, scale: float, simulated: bool) -> str:
    if not placebos:
        return ""
    header = " & ".join(
        [
            "检验" if language == "zh" else "Test",
            "Draws" if language == "en" else "抽取",
            "实际均值" if language == "zh" else "Actual",
            "空值均值" if language == "zh" else "Null mean",
            "P5",
            "P95",
            "百分位" if language == "zh" else "Percentile",
            "p",
        ]
    )
    names = {"in_time": text(language, "in_time"), "in_space": text(language, "in_space")}
    body = []
    for key, result in placebos.items():
        body.append(
            " & ".join(
                [
                    names.get(key, key),
                    str(result["draws"]),
                    _num(result["actual"] * scale),
                    _num(result["null_mean"] * scale),
                    _num(result["p05"] * scale),
                    _num(result["p95"] * scale),
                    _num(result.get("percentile", np.nan), 1),
                    _num(result["p_value"], 3),
                ]
            )
            + r" \\"
        )
    tabular = "\n".join(
        [
            "\\begin{tabular}{lrrrrrrr}",
            "\\toprule",
            header + r" \\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    caption = "安慰剂检验" if language == "zh" else "Placebo tests"
    if language == "zh":
        notes = [
            "时间安慰剂把每家公司的事件日移到估计窗之前的其他交易日。",
            "截面安慰剂在样本内重配价格序列，事件日不变。",
            "两边的 p 值都是双侧随机化 p，加一修正。它们不是处理效应的 p 值。",
            "百分位是实际平均 CAR 在安慰剂抽取里的经验分布位置。时间安慰剂会把估计窗一起挪到伪事件日之前再重新拟合。",
        ]
    else:
        notes = [
            "The in-time placebo moves each firm to another pre-event trading day.",
            "The in-space placebo reassigns price series inside the sample and keeps the event dates.",
            "Both p-values are two-sided randomization p-values with an add-one correction. They are not ATT p-values.",
            "The percentile is the empirical rank of the actual mean CAR among the placebo draws. In-time draws refit the estimation window at the pseudo-event.",
        ]
    if simulated:
        notes.insert(0, text(language, "simulated"))
    notes.append(text(language, "scale_note", scale=scale))
    return _wrap_table(caption, "tab:placebo", tabular, notes)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
