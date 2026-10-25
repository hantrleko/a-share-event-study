"""Methods paragraphs generated from the run that just finished.

The text describes the statistical benchmark. It does not claim identification.
"""

from __future__ import annotations

from ashare_event.config import StudyConfig, window_label

_MODELS = {
    "zh": {
        "market": "市场模型：在估计窗内为每家公司拟合「个股收益 = 截距 + 斜率 × 基准收益」",
        "market_adjusted": "市场调整模型：事件窗收益直接减去基准收益，不估计截距和斜率",
        "industry": "行业基准的市场模型：斜率对着使用者提供的行业指数，而不是宽基指数",
        "ff3": "三因子模型：截距加上使用者提供的 mkt、smb 和 hml",
    },
    "en": {
        "market": "a market model fitted in the estimation window",
        "market_adjusted": "a market-adjusted model that subtracts the benchmark return",
        "industry": "a market model whose benchmark is a user-supplied industry index",
        "ff3": "a three-factor model using user-supplied mkt, smb, and hml",
    },
}


def render_methods(
    cfg: StudyConfig,
    *,
    n_events: int,
    n_firms: int,
    preferred: str,
    r_bar: float,
    simulated: bool,
) -> tuple[str, str]:
    windows = "、".join(window_label(*item) for item in cfg.event_windows)
    pref = {"portfolio": "组合时间序列", "kp": "Kolari–Pynnönen 调整后的 BMP", "bmp": "BMP"}.get(preferred, preferred)
    pref_en = {"portfolio": "the portfolio time-series test", "kp": "the Kolari-Pynnonen adjusted BMP", "bmp": "BMP"}.get(
        preferred, preferred
    )
    sim_zh = "本段里的样本来自模拟路径，数字不能写入论文。" if simulated else "事件日由使用者提供，程序不自行认定处理组和对照组。"
    sim_en = "This run uses simulated paths and its numbers are not empirical evidence." if simulated else (
        "Event dates are supplied by the user. The program does not construct a control group."
    )
    zh = "\n".join(
        [
            sim_zh,
            (
                f"估计样本有 {n_events} 条公司-事件、{n_firms} 家公司。"
                "非交易日向后 10 个日历日内对齐到基准指数的第一个交易日；"
                "若事件表把 after_close 标成盘后，则从公告日的下一天开始对齐。"
                "事件窗里缺一个交易日，该窗口的累计异常收益作废。"
            ),
            (
                f"收益是简单收益率。模型是{_MODELS['zh'].get(cfg.model, cfg.model)}。"
                "行业指数和三因子都由使用者提供，程序不下载、也不自行匹配申万行业或因子库。"
                f"估计窗为 {window_label(*cfg.estimation_window)}，必须整段落在路径窗 {window_label(*cfg.path_window)} "
                f"和报告窗 {windows} 的左边。回归窗是 {window_label(*cfg.regression_window)}。"
            ),
            (
                "平均累计异常收益对每条公司-事件等权。事件日有一半以上落在同一天时，首选检验是等权组合的时间序列 t；"
                f"否则若估计窗残差平均相关高于 0.05，首选为 KP；其余为 BMP。这一次回归窗的首选是{pref}"
                f"（r_bar = {r_bar:.3f}）。横截面 t 和秩检验只作对照。"
                "多个窗口的星号没有做多重检验校正；p BH 只对表内这些窗口的首选 p 值做 Benjamini–Hochberg 调整。"
            ),
            (
                "时间安慰剂把事件日挪到估计窗之前的其他交易日，并在伪事件日上重新拟合估计窗，不是把真实事件的异常收益平移。"
                "截面安慰剂在样本内重配价格序列后重新拟合。两侧都用加一修正的双侧随机化 p。"
                "百分位只说明真实日期在伪日期分布里的位置。安慰剂不是对照组。"
            ),
            (
                f"横截面回归的标准误按 {cfg.cluster} 聚类。"
                "双向聚类列同时使用公司和事件日。聚类数较少时，快照里另有 wild cluster bootstrap p，星号仍来自聚类标准误。"
                "系数是累计异常收益和公司特征的相关。市场模型异常收益不是平均处理效应。"
            ),
            "事件日若靠近按板块推断的涨跌停带宽，会在样本日志里计数。这是启发式标记，不是交易所的涨跌停状态表，程序不因此删行。",
        ]
    )
    en = "\n".join(
        [
            sim_en,
            (
                f"The sample has {n_events} firm-events and {n_firms} firms. "
                "A non-trading announcement maps to the next benchmark trading day within 10 calendar days. "
                "after_close starts that search on the following day. A missing day inside a CAR window drops that CAR."
            ),
            (
                f"Returns are simple returns. The benchmark is {_MODELS['en'].get(cfg.model, cfg.model)}. "
                "Industry indexes and factors are supplied by the user; the program does not download them. "
                f"The estimation window is {window_label(*cfg.estimation_window)}, strictly before the path "
                f"{window_label(*cfg.path_window)} and the reported windows {windows}. "
                f"The regression window is {window_label(*cfg.regression_window)}."
            ),
            (
                "Mean CAR weights each firm-event equally. The preferred test is the portfolio time-series t when "
                "at least half the events share a date, the Kolari-Pynnonen adjusted BMP when average residual "
                f"correlation exceeds 0.05, and BMP otherwise. This run prefers {pref_en} (r_bar = {r_bar:.3f}). "
                "The cross-sectional t and the rank t are reported alongside. Stars are not multiplicity-adjusted; "
                "p BH is a Benjamini-Hochberg adjustment of the preferred p-values in the window table."
            ),
            (
                "The in-time placebo moves each event onto an earlier trading day and refits the estimation window there. "
                "The in-space placebo reassigns price series and refits. Both use an add-one two-sided randomization p-value. "
                "The percentile locates the actual mean in that pseudo-date distribution. Neither placebo is a control group."
            ),
            (
                f"Cross-sectional standard errors are clustered by {cfg.cluster}. "
                "Two-way columns also cluster by event date. When there are few clusters, the manifest may include "
                "wild cluster bootstrap p-values; stars still use the clustered standard errors. "
                "Coefficients are associations. A market-model abnormal return is not an average treatment effect."
            ),
            "Days close to a heuristic price-limit band are counted in the sample log and are not dropped.",
        ]
    )
    return zh + "\n", en + "\n"
