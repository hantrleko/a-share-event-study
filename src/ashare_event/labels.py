"""User-facing labels. Tables follow config.language; keys stay in English."""

from __future__ import annotations

STRINGS = {
    "zh": {
        "rel_day": "相对事件日的交易日",
        "mean_car": "平均累计异常收益",
        "placebo_x": "安慰剂平均累计异常收益",
        "placebo_y": "抽取次数",
        "in_time": "时间安慰剂",
        "in_space": "截面安慰剂",
        "window": "窗口",
        "n": "N",
        "mean": "平均 CAR",
        "t_cs": "t 横截面",
        "t_bmp": "t BMP",
        "t_kp": "t KP",
        "t_port": "t 组合",
        "preferred": "首选",
        "pref_portfolio": "组合",
        "pref_kp": "KP",
        "pref_bmp": "BMP",
        "intercept": "常数项",
        "signal_cost": "信号成本",
        "size_log": "规模（对数）",
        "legitimacy_gap": "合法性缺口",
        "peer_count_pm5": "同业事件数（±5 日）",
        "fe_industry": "行业固定效应",
        "fe_year": "年份固定效应",
        "r2": "R²",
        "r2_within": "组内 R²",
        "clusters": "聚类数",
        "yes": "是",
        "no": "否",
        "simulated": "模拟数据，不能作为实证证据。",
        "not_att": "市场模型异常收益是相对基准的统计偏离，不是平均处理效应。",
        "stars": "*、**、*** 分别表示 10%、5%、1% 的显著性。",
        "cluster_note": "回归标准误按 {cluster} 聚类。",
        "scale_note": "收益乘以 {scale:g} 后制表，因此 1 表示 1 个百分点（当 scale 为 100）。",
        "preferred_note": "平均 CAR 的星号来自首选检验，而不是横截面 t。横截面 t 在事件日重合或残差相关时会过高估计显著性。",
        "event_types": {
            "ai_disclosure": "AI 披露",
            "ai_policy": "人工智能政策",
            "esg": "ESG",
            "esg_guideline": "ESG 指引",
            "geopolitics": "地缘政治",
            "unspecified": "未分类",
        },
    },
    "en": {
        "rel_day": "Trading days relative to the event",
        "mean_car": "Mean cumulative abnormal return",
        "placebo_x": "Placebo mean CAR",
        "placebo_y": "Draws",
        "in_time": "In-time placebo",
        "in_space": "In-space placebo",
        "window": "Window",
        "n": "N",
        "mean": "Mean CAR",
        "t_cs": "t CS",
        "t_bmp": "t BMP",
        "t_kp": "t KP",
        "t_port": "t portfolio",
        "preferred": "Preferred",
        "pref_portfolio": "portfolio",
        "pref_kp": "KP",
        "pref_bmp": "BMP",
        "intercept": "Constant",
        "signal_cost": "Signal cost",
        "size_log": "Log size",
        "legitimacy_gap": "Legitimacy gap",
        "peer_count_pm5": "Peer events (±5 days)",
        "fe_industry": "Industry FE",
        "fe_year": "Year FE",
        "r2": "R²",
        "r2_within": "Within R²",
        "clusters": "Clusters",
        "yes": "Yes",
        "no": "No",
        "simulated": "SIMULATED DATA, not empirical evidence.",
        "not_att": "Market-model abnormal returns are a statistical benchmark, not an average treatment effect.",
        "stars": "* p<0.10, ** p<0.05, *** p<0.01.",
        "cluster_note": "Standard errors are clustered by {cluster}.",
        "scale_note": "Returns are multiplied by {scale:g} before display.",
        "preferred_note": "Stars on mean CAR use the preferred test, not the cross-sectional t. The cross-sectional t overstates significance when event dates coincide or residuals are correlated.",
        "event_types": {
            "ai_disclosure": "AI disclosure",
            "esg": "ESG",
            "geopolitics": "Geopolitics",
            "unspecified": "Unspecified",
        },
    },
}

COEF_KEYS = ("Intercept", "signal_cost", "size_log", "legitimacy_gap", "peer_count_pm5")


def text(language: str, key: str, **fmt) -> str:
    bundle = STRINGS.get(language, STRINGS["zh"])
    value = bundle[key]
    return value.format(**fmt) if fmt else value


def coef_label(language: str, name: str) -> str:
    bundle = STRINGS.get(language, STRINGS["zh"])
    if name == "Intercept":
        return bundle["intercept"]
    return bundle.get(name, name)


def event_type_label(language: str, name: str) -> str:
    bundle = STRINGS.get(language, STRINGS["zh"])
    return bundle["event_types"].get(name, name)
