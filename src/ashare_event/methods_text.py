"""Methods paragraphs generated from the run that just finished.

The text describes the statistical benchmark. It does not claim identification.
"""

from __future__ import annotations

from ashare_event.config import StudyConfig, window_label

_MODELS = {
    "zh": {
        "market": "市场模型：在估计窗内为每家公司拟合「个股收益 = 截距 + 斜率 × 基准收益」",
        "market_adjusted": "市场调整模型：事件窗收益直接减去基准收益，不估计截距和斜率",
        "industry": "行业基准的市场模型：斜率对着行业指数，而不是宽基指数",
        "ff3": "三因子模型：截距加上 mkt、smb 和价值因子",
        "ff5": "五因子模型：在三因子之外加上使用者提供的 rmw 和 cma",
        "carhart": "Carhart 四因子：在三因子之外加上使用者提供的 umd",
    },
    "en": {
        "market": "a market model fitted in the estimation window",
        "market_adjusted": "a market-adjusted model that subtracts the benchmark return",
        "industry": "a market model whose benchmark is an industry index",
        "ff3": "a three-factor model using mkt, smb, and a value factor",
        "ff5": "a five-factor model that also uses user-supplied rmw and cma",
        "carhart": "a Carhart model that also uses a user-supplied umd factor",
    },
}


def _benchmark_zh(cfg: StudyConfig) -> str:
    industry = cfg.extra.get("industry_source")
    factor = cfg.extra.get("factor_source")
    if industry == "shenwan_l1":
        return "行业指数是申万一级，按分类历史在事件日有效的一级行业匹配。交易日仍按宽基指数。事件表里的行业名称只用于筛选。"
    if factor == "ch3":
        return (
            "三因子是 Liu-Stambaugh-Yuan CH3 日频的市场超额收益、SMB 和 VMG。VMG 存在 hml 列，不是 Fama-French 的 HML。"
            "个股收益先减去无风险利率。五因子的 RMW、CMA 和 Carhart 的 UMD 不在这份库里。"
        )
    if cfg.model in {"industry", "ff3", "ff5", "carhart"}:
        return "行业指数或因子由使用者上传。五因子需要 RMW、CMA，Carhart 需要 UMD。"
    return "基准默认是沪深 300。"


def _benchmark_en(cfg: StudyConfig) -> str:
    industry = cfg.extra.get("industry_source")
    factor = cfg.extra.get("factor_source")
    if industry == "shenwan_l1":
        return "The industry index is Shenwan level 1, matched from the classification in effect on the event date. "
    if factor == "ch3":
        return "The three factors are Liu-Stambaugh-Yuan CH3: market excess, SMB, and VMG. Stock returns are in excess of the risk-free rate. "
    if cfg.model in {"industry", "ff3", "ff5", "carhart"}:
        return "Industry indexes or factors are supplied by the user. "
    return "The default benchmark is the CSI 300. "


def _limit_zh(cfg: StudyConfig) -> str:
    if cfg.drop_event_day_limit:
        return (
            "事件日涨跌停的公司-事件已从 CAR 样本剔除。交易所涨停池不含 ST 和科创板，跌停池只覆盖近 30 日；"
            "名单没有覆盖到的，用板块和公司名称推断的带宽。"
        )
    return "事件日若靠近按板块推断的涨跌停带宽，会在样本日志里计数。默认不因此删行。"


def _limit_en(cfg: StudyConfig) -> str:
    if cfg.drop_event_day_limit:
        return (
            "Firm-events with a limit hit on the event day are dropped. "
            "The exchange limit-up list omits ST and STAR names, and the limit-down list covers about 30 days; "
            "other names use the board band."
        )
    return "Days close to a heuristic price-limit band are counted in the sample log and are not dropped."


def _supplement_zh(cfg: StudyConfig) -> str:
    parts = [
        "BHAR 与 CAR 分列存放，不互相替代。日历时间 t、广义符号 z 和 KP 调整后的秩检验写在补充表里，不改变首选规则，也不进入 BH。"
    ]
    if cfg.anticipation_window is not None:
        parts.append(
            f"预事件窗 {window_label(*cfg.anticipation_window)} 单独计算，不因此剔除事件。"
        )
    if float(cfg.winsorize or 0) > 0:
        parts.append("横截面 CAR 和 BHAR 做了缩尾。安慰剂对照的是缩尾前的平均 CAR。路径没有缩尾。")
    if int(cfg.market_lags or 0):
        parts.append(
            "估计窗加入市场收益的前一日和后一日，斜率相加。事件窗预测只用同期市场收益。方差仍按单指数公式，这是近似。"
        )
    if cfg.extra.get("signal_source") == "mda_digital_per_k":
        parts.append(
            "信号成本一栏来自年报文本特征引擎的管理层讨论数智化或人工智能用词，单位是每千词。"
            "同一引擎的绿色/ESG 与地缘每千词各自成列，不写入合法性缺口。"
            "这些都是披露用词强度，不是信号成本、ESG 表现或地缘冲击本身，系数仍是相关。"
        )
    return "".join(parts)


def _supplement_en(cfg: StudyConfig) -> str:
    parts = [
        "BHAR is stored beside CAR. Calendar-time t, the generalized sign z, and the KP-adjusted rank t are supplements. "
        "They do not change the preferred test or enter the BH adjustment."
    ]
    if cfg.anticipation_window is not None:
        parts.append(
            f" The pre-event window {window_label(*cfg.anticipation_window)} is reported separately and does not drop events."
        )
    if float(cfg.winsorize or 0) > 0:
        parts.append(" Cross-sectional CAR and BHAR are winsorized. Placebos use the pre-winsor mean. The path is not winsorized.")
    if int(cfg.market_lags or 0):
        parts.append(
            " The estimation window adds one market lead and one lag, and the slopes are summed. "
            "The event-window forecast uses only the same-day market return. Variance stays the single-index formula."
        )
    if cfg.extra.get("signal_source") == "mda_digital_per_k":
        parts.append(
            " The signal-cost column is the annual-report text engine's MD&A digital or AI words per thousand tokens."
            " That engine's ESG and geopolitical intensities stay in their own columns."
            " All three are disclosure-word intensities, not measured signal cost, ESG performance, or a geopolitical shock."
            " The coefficient remains an association."
        )
    return "".join(parts)


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
                f"{_benchmark_zh(cfg)}"
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
            _limit_zh(cfg),
            _supplement_zh(cfg),
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
                f"{_benchmark_en(cfg)}"
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
            _limit_en(cfg),
            _supplement_en(cfg),
        ]
    )
    return zh + "\n", en + "\n"
