"""Pre-estimation mechanism notes for three lenses.

This module is a design aid. It does not confirm a mechanism, and it must
not be pasted into a paper as a finding. A post-estimation write-up has to
start from the whole result pattern, including the nulls.

Lenses follow the usual primitives: Spence (1973) for signaling, DiMaggio
and Powell (1983) for institutional isomorphism, Suchman (1995) for
legitimacy. Those citations name the benchmark stories. This run did not
re-search the literature, and it did not check page numbers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LensNote:
    key: str
    name_zh: str
    name_en: str
    chain_zh: str
    chain_en: str
    benchmark_zh: str
    benchmark_en: str
    predictions_zh: tuple[str, ...]
    predictions_en: tuple[str, ...]
    reject_zh: tuple[str, ...]
    reject_en: tuple[str, ...]
    boundary_zh: str
    boundary_en: str


SIGNALING = LensNote(
    key="signaling",
    name_zh="信号",
    name_en="Signaling",
    chain_zh="信息不对称加深或出现新的不可观察类型 → 只有高成本披露才能把类型分开 → 可信信号对应正的异常收益，廉价表态对应接近零的异常收益。",
    chain_en="A wider information gap → only a costly disclosure separates types → a credible signal lines up with a positive abnormal return, and cheap talk lines up with a return near zero.",
    benchmark_zh="基准是 Spence (1973) 的分离均衡。相对这个基准，经验上要检验的不是“披露日有没有反应”，而是反应是否随信号成本上升，并在成本消失时消失。",
    benchmark_en="The benchmark is the separating equilibrium in Spence (1973). Relative to that benchmark, the empirical question is not whether the disclosure day moves; it is whether the move rises with signal cost and disappears when the cost disappears.",
    predictions_zh=(
        "回归表里，信号成本的系数为正。成本从低到高，累计异常收益应更高。",
        "披露一旦变成强制、统一、文本高度同质，这条斜率应变得更平。",
        "第三方核验、可核对的细节、不可轻易撤回的承诺，应比口号式表述对应更大的反应。",
    ),
    predictions_en=(
        "In the regression table, the signal-cost coefficient is positive.",
        "The slope flattens when disclosure is mandatory, uniform, and boilerplate.",
        "Verification, checkable detail, and hard-to-reverse commitments should move prices more than slogans.",
    ),
    reject_zh=(
        "信号成本更高的公司，累计异常收益并不更高。",
        "低成本、可复制的文本反而对应更大的正反应。",
    ),
    reject_en=(
        "Firms with higher signal cost do not have higher CARs.",
        "Low-cost, easy-to-copy text lines up with the larger positive reaction.",
    ),
    boundary_zh="若披露和订单、罚款、补贴或资本开支绑在同一天公布，价格里同时有现金流消息。这时正的累计异常收益不能写成信号本身。",
    boundary_en="If the disclosure arrives with an order, a fine, a subsidy, or a capital-spending item, the price also contains a cash-flow message. A positive CAR is then not the signal itself.",
)

LEGITIMACY = LensNote(
    key="legitimacy",
    name_zh="合法性",
    name_en="Legitimacy",
    chain_zh="公司与受众之间出现合法性缺口 → 公司用补救或披露试图修复实用、道德或认知上的接受 → 受众把实质性补救和印象管理分开，缺口越大，负面事件的价格惩罚越重。",
    chain_en="A legitimacy gap opens between the firm and its audience → the firm tries to repair pragmatic, moral, or cognitive acceptance → the audience separates substantive repair from impression management, so a larger gap lines up with a heavier price penalty.",
    benchmark_zh="基准是 Suchman (1995) 对合法性管理的区分。和现金流渠道比，这条机制要求：在可观察的经营损失之外，缺口仍然解释残差。做不到这一点，就只能把它留在竞争解释里。",
    benchmark_en="The benchmark is Suchman's (1995) distinction among ways of managing legitimacy. Against a cash-flow channel, the mechanism requires that the gap still explains residual returns after observable operating losses. If it does not, keep it as a competing account.",
    predictions_zh=(
        "回归表里，合法性缺口的系数为负。",
        "事件本身就是罚款、丢单或制裁时，先写出现金流和风险渠道。合法性只解释扣掉这些暴露之后剩下的折价。",
        "事前记录与事后补救一致的公司，下跌应更轻；记录相反时，补救更像印象管理，折价不应消失。",
    ),
    predictions_en=(
        "In the regression table, the legitimacy-gap coefficient is negative.",
        "When the event is a fine, a lost contract, or a sanction, write the cash-flow and risk channel first. Legitimacy is a claim about the discount that remains after that exposure.",
        "Firms whose past record matches the remedy should fall less. A contradictory record should not be fully repaired by the announcement.",
    ),
    reject_zh=(
        "控制可观察的经营暴露之后，合法性缺口没有额外的解释力。",
        "缺口更大的公司，负向反应反而更弱。",
    ),
    reject_en=(
        "After observable operating exposure is held fixed, the legitimacy gap adds nothing.",
        "Firms with a larger gap have the weaker negative reaction.",
    ),
    boundary_zh="合同、抵押、担保或已经很透明的报表若已经把支持写死，合法性故事的边际应更弱。",
    boundary_en="The legitimacy margin should be weaker when contracts, collateral, guarantees, or already-transparent accounts already pin down support.",
)

INSTITUTIONAL = LensNote(
    key="institutional",
    name_zh="制度同构",
    name_en="Institutional isomorphism",
    chain_zh="监管、规范压力或同业实践改变了“合格公司应当做什么” → 公司被迫遵从、互相模仿，或按规范采纳；采纳可以是实质的，也可以和内部做法脱耦 → 仪式性扎堆的平均异常收益接近零，传递新约束的扎堆则为负。",
    chain_en="Regulation, normative pressure, or peer practice changes what a legitimate firm is expected to do → firms comply, imitate, or adopt ceremonially → ceremonial clustering lines up with a mean abnormal return near zero, and clustering that reveals a new constraint lines up with a negative return.",
    benchmark_zh="基准是 DiMaggio and Powell (1983) 的三种同构。和信号理论比，同构不要求成本排序：大家可以在同一周做同一件事，而价格几乎不动。同构理论本身不决定收益的符号。",
    benchmark_en="The benchmark is the three forms of isomorphism in DiMaggio and Powell (1983). Against signaling, isomorphism does not need a cost ranking: many firms can do the same thing in the same week and the price can barely move. The theory does not by itself pin down the sign.",
    predictions_zh=(
        "把 peer_count_pm5 放进回归后，若系数接近零且平均累计异常收益也接近零，更符合仪式性遵从。",
        "若同业扎堆伴随负的累计异常收益，更符合市场在给新的合规成本或新约束定价，而不是“披露本身带来合法性溢价”。",
        "独立、不可被同业日历解释的披露若呈现稳定正溢价，同构就不是主渠道。",
    ),
    predictions_en=(
        "Once peer_count_pm5 is in the regression, a slope near zero together with a mean CAR near zero fits ceremonial compliance.",
        "A negative CAR where peers cluster fits a newly priced constraint, not a legitimacy premium for disclosing.",
        "A stable positive premium on disclosures that are not explained by the peer calendar cuts against isomorphism as the main channel.",
    ),
    reject_zh=(
        "扎堆程度和累计异常收益的关系既不是零，也不是负，而是稳定为正。",
        "事件日高度重合，但组合时间序列检验把横截面上的“显著”吃掉。这时看到的是共同冲击，不是同构。",
    ),
    reject_en=(
        "The relation between clustering and CAR is a stable positive slope, not zero and not negative.",
        "Event dates coincide and the portfolio time-series test removes the cross-sectional significance. That is a common shock, not isomorphism.",
    ),
    boundary_zh="peer_count_pm5 只数估计样本里的同业事件，不是全市场的同业采纳，而且它是内生的。它不能单独当作识别。",
    boundary_en="peer_count_pm5 counts peer events inside the estimation sample, not market-wide adoption, and it is endogenous. It is not an identification strategy on its own.",
)

LENSES = {
    "signaling": SIGNALING,
    "legitimacy": LEGITIMACY,
    "institutional": INSTITUTIONAL,
}

PRIMARY = {
    "ai_disclosure": ("signaling", "legitimacy", "institutional"),
    "esg": ("legitimacy", "institutional", "signaling"),
    "geopolitics": ("institutional", "legitimacy", "signaling"),
    "unspecified": ("signaling", "legitimacy", "institutional"),
}

COLUMN_MAP = {
    "signal_cost": ("signaling", "信号成本", "signal cost"),
    "legitimacy_gap": ("legitimacy", "合法性缺口", "legitimacy gap"),
    "peer_count_pm5": ("institutional", "同业事件数", "peer-event count"),
    "size_log": ("", "规模", "size"),
}


def _pick(note: LensNote, language: str, field: str):
    return getattr(note, f"{field}_{language if language in {'zh', 'en'} else 'zh'}")


def render_theory(event_types: list[str], columns: list[str], language: str = "zh") -> str:
    language = "en" if language == "en" else "zh"
    present = [item for item in event_types if item] or ["unspecified"]
    order: list[str] = []
    for event_type in present:
        for key in PRIMARY.get(event_type, PRIMARY["unspecified"]):
            if key not in order:
                order.append(key)
    if language == "zh":
        return _render_zh(present, columns, order)
    return _render_en(present, columns, order)


def _render_zh(present: list[str], columns: list[str], order: list[str]) -> str:
    lines = [
        "# 候选机制备忘",
        "",
        "状态：估计前的候选记录。这里没有识别任何机制，也不能把显著系数写成机制已经被证实。",
        "估计做完之后，要连同不显著和与预期相反的结果一起重写；不能只围绕显著的那一列补故事。",
        "",
        "## 这次样本里的事件",
        "",
        "、".join(present) + "。",
        "",
    ]
    if "geopolitics" in present:
        lines.extend(
            [
                "## 地缘政治事件先写哪条渠道",
                "",
                "地缘政治冲击的默认渠道是预期现金流和风险，不是信号、合法性或同构。",
                "制裁、出口管制、冲突升级，先改变订单、成本和折现率。",
                "制度同构只在问题变成“监管约束如何落到一类公司”时才进入。",
                "合法性只在经营暴露被扣掉之后，仍要解释剩余的声誉折价时才进入。",
                "除非事件本身是公司对敞口的披露，否则信号透镜不适合当主故事。",
                "",
            ]
        )
    if "ai_disclosure" in present:
        lines.extend(
            [
                "## AI 披露先比较哪两条",
                "",
                "主候选是信号：市场想区分真有能力的披露和廉价的 AI 表态。",
                "合法性是竞争解释：披露也可能是在回应信任危机，而不是在报告能力。",
                "同构是第三条：第一家公司披露之后，同业在随后几天跟进，跟进本身可以没有信息含量。",
                "",
            ]
        )
    if "esg" in present:
        lines.extend(
            [
                "## ESG 事件先比较哪两条",
                "",
                "主候选是合法性：负面 ESG 事件惩罚缺口，补救行动是否被定价取决于受众是否把它当成实质修复。",
                "同构是竞争解释：强制 ESG 披露可以是仪式性遵从。",
                "信号只在核验或承诺真的有成本时才站得住。没有成本的 ESG 句子不能分开类型。",
                "",
            ]
        )
    lines.extend(["## 三条透镜", ""])
    for key in order:
        note = LENSES[key]
        lines.extend(
            [
                f"### {note.name_zh}",
                "",
                "核心机制：" + note.chain_zh,
                "",
                "对照的基准：" + note.benchmark_zh,
                "",
                "最小骨架（未求解）：公司管理者面对投资者或受众；事件日更新信息；价格反映受众是否相信类型、修复或遵从。这里没有解出均衡，也没有比较静态的闭式结果。",
                "",
                "候选命题：",
                *[f"- {item}" for item in note.predictions_zh],
                "",
                "比较静态：" + note.boundary_zh,
                "",
                "什么结果会拒绝它：",
                *[f"- {item}" for item in note.reject_zh],
                "",
            ]
        )
    lines.extend(["## 和表里的变量", ""])
    for column, (lens, zh, _en) in COLUMN_MAP.items():
        if column in columns:
            owner = LENSES[lens].name_zh if lens in LENSES else "控制变量"
            lines.append(f"- `{column}` 已在事件表中，对应{owner}（{zh}）。")
        else:
            lines.append(f"- `{column}` 还没有进入事件表。相关预测先不要写进正文。")
    lines.extend(
        [
            "",
            "规模是控制变量，不是机制。",
            "同业事件数默认不进回归。需要制度透镜的那一列时，在配置里把 `include_peer_count` 设为 true，并在表注里写明它内生。",
            "",
            "## 可放进论文的段落",
            "",
            _paragraph_zh(present),
            "",
            "## 不要写的句子",
            "",
            "- 不要写“事件研究识别了信号/合法性/制度效应”。",
            "- 不要写“显著的交互项证明了机制”。",
            "- 不要把安慰剂 p 值写成因果 p 值。",
            "- 不要在估计前把上面的候选段落放进结论。",
            "",
            "## 文献位置",
            "",
            "Spence (1973)、DiMaggio and Powell (1983)、Suchman (1995) 只用来命名三条基准故事。",
            "异常收益的统计基准是 MacKinlay (1997)；事件窗方差用 Boehmer, Musumeci, and Poulsen (1991)；横截面相关用 Kolari and Pynnonen (2010)。",
            "本次运行没有重新检索文献，没有核对页码，也没有声称这些机制相对 A 股文献是新的。",
            "",
        ]
    )
    return "\n".join(lines)


def _paragraph_zh(present: list[str]) -> str:
    if present == ["geopolitics"] or (len(present) == 1 and present[0] == "geopolitics"):
        return (
            "一种尚未经结果约束的读法是：地缘政治事件首先改写预期现金流和风险。"
            "只有在经营暴露被单独记清之后，仍剩下的折价才可能和合法性缺口有关；"
            "只有在事件是一类公司共同面对的新约束时，制度压力才进入解释。"
            "若组合时间序列检验吸收了横截面上的显著，优先报告共同冲击，而不是社会理论。"
        )
    return (
        "一种尚未经结果约束的读法是：披露要有价格含义，需要受众能把它和廉价表态分开。"
        "信号成本更高、更可核对的披露，累计异常收益应更高；只是追随同业的披露，均值应接近零；"
        "用来填补信任缺口的补救，则应在缺口更大时表现出更重的折价，而不是自动的正溢价。"
        "这三句是估计前的可区分预测。哪一句留下来，取决于完整的结果，包括不显著的列。"
    )


def _render_en(present: list[str], columns: list[str], order: list[str]) -> str:
    lines = [
        "# Candidate mechanisms",
        "",
        "Status: pre-estimation note. Nothing here is an identified mechanism, and a significant coefficient is not confirmation.",
        "After estimation, rewrite from the full pattern, including nulls and results that cut against the story.",
        "",
        "## Event types in this sample",
        "",
        ", ".join(present) + ".",
        "",
    ]
    if "geopolitics" in present:
        lines.extend(
            [
                "## What comes first for a geopolitical event",
                "",
                "The default channel is expected cash flow and risk, not signaling, legitimacy, or isomorphism.",
                "Sanctions, export controls, and escalation move orders, costs, and discount rates.",
                "Isomorphism enters only when the question is how a new rule hits a class of firms.",
                "Legitimacy enters only as a residual reputation discount after operating exposure is accounted for.",
                "Signaling is a poor lead unless the event is the firm's own disclosure of exposure.",
                "",
            ]
        )
    lines.extend(["## Three lenses", ""])
    for key in order:
        note = LENSES[key]
        lines.extend(
            [
                f"### {note.name_en}",
                "",
                "Core mechanism: " + note.chain_en,
                "",
                "Benchmark contrast: " + note.benchmark_en,
                "",
                "Minimal skeleton (not solved): a manager faces investors or an audience; the event updates information; the price reflects whether the audience believes the type, the repair, or the compliance. No equilibrium is solved here.",
                "",
                "Candidate propositions:",
                *[f"- {item}" for item in note.predictions_en],
                "",
                "Boundary: " + note.boundary_en,
                "",
                "What would reject it:",
                *[f"- {item}" for item in note.reject_en],
                "",
            ]
        )
    lines.extend(["## Columns", ""])
    for column, (lens, _zh, en) in COLUMN_MAP.items():
        if column in columns:
            owner = LENSES[lens].name_en if lens in LENSES else "a control"
            lines.append(f"- `{column}` is on the event list and maps to {owner} ({en}).")
        else:
            lines.append(f"- `{column}` is not on the event list. Do not put that prediction in the text yet.")
    lines.extend(
        [
            "",
            "Size is a control, not a mechanism.",
            "The peer count stays out of the regression unless `include_peer_count` is true. If it is included, the note must say it is endogenous.",
            "",
            "## Paragraph that can go in a paper, with the boundary left visible",
            "",
            (
                "One reading, not yet disciplined by the results, is that a disclosure moves the price only when the audience can separate it from cheap talk. "
                "A costlier, checkable disclosure should have a higher CAR; a disclosure that only follows peers should have a mean near zero; "
                "a remedy aimed at a trust gap should show a heavier discount when the gap is larger, not an automatic premium. "
                "These are pre-estimation contrasts. Which sentence survives depends on the full pattern, including the nulls."
            ),
            "",
            "## Sentences to avoid",
            "",
            "- Do not write that the event study identifies a signaling, legitimacy, or institutional effect.",
            "- Do not write that a significant interaction proves the mechanism.",
            "- Do not describe a placebo p-value as a causal p-value.",
            "",
            "## Literature placement",
            "",
            "Spence (1973), DiMaggio and Powell (1983), and Suchman (1995) name the three benchmark stories.",
            "The statistical benchmark for abnormal returns is MacKinlay (1997); event-window variance follows Boehmer, Musumeci, and Poulsen (1991); cross-sectional correlation follows Kolari and Pynnonen (2010).",
            "This run did not re-search the literature, did not check page numbers, and does not claim that the mechanisms are new relative to the A-share literature.",
            "",
        ]
    )
    return "\n".join(lines)
