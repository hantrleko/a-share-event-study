# A 股事件研究流水线

换一张事件表，就能重算累计异常收益、路径图、安慰剂和横截面回归表。演示样本可以立刻跑通；真实论文换成自己的事件日，并用 AKShare 或本地行情。

市场模型异常收益是相对基准的统计偏离，不是平均处理效应。回归系数是相关关系。信号、合法性、制度同构写在估计前的候选备忘里，不能当成已经识别的机制。

## 在线使用

打开 [事件研究网站](https://a-share-event-study-production.up.railway.app)。当前版本 V0.2。演示样本不需要安装。源码在 [hantrleko/a-share-event-study](https://github.com/hantrleko/a-share-event-study)。

## 先跑演示

在项目目录打开 PowerShell：

```powershell
.\run_demo.ps1
```

等价命令是 `python -m ashare_event demo`，需要 `PYTHONPATH=src`。演示不访问网络。终端里的数字和 `paper/tabs`、`paper/figs` 都来自模拟路径，文件里会标明，不能写进论文。

交互筛选：

```powershell
$env:PYTHONPATH = "src"
streamlit run app/streamlit_app.py
```

侧栏可选行业、事件类型和 CAR 窗口。图仍然用配置里的路径窗（默认 [-10, 10]）。回归、检验和安慰剂用侧栏选定的窗口。

## 换成自己的事件

交互页左侧就是全部输入。事件表的标准列是固定的：证券代码、事件日期、事件类型、行业、公司名称。要放进回归的量在左侧「加入变量」，会变成同一张表里的新列。已有 CSV 只要用这些标准列名，或常见中文列名（证券代码、公告日期、事件类型、行业），就可以导入。

切到「我的事件表」后，可以下载带两套示例事件的 CSV，或只下载当前表头。示例是政策发布日，公司名带「示例」，AKShare 会拒绝下载。核对后再删掉这两个字。命令行仍可读 `config/study.yml`。

表在 `paper/tabs/*.tex`，图在 `paper/figs/*.pdf`。论文里 `\input` 这些 tex，不要手抄数字。方法段骨架在 `paper/snippets/methods_skeleton.tex`。样本怎么被筛掉，写在 `output/logs/sample_log.md`。

## 默认设定

这些选择写在 `config/study.yml`，改配置即可，不必改代码。

| 项目 | 默认 | 要改时 |
| --- | --- | --- |
| 估计对象 | 市场模型累计异常收益 | 不是处理效应 |
| 模型 | 估计窗内的市场模型 | `model: market_adjusted` 则把基准收益直接扣掉 |
| 估计窗 | [-250, -31] | 必须整段落在路径窗和所有 CAR 窗之前 |
| 报告窗 | [-1, 1]、[-3, 3]、[-5, 5] | 回归默认用 [-1, 1] |
| 路径图 | [-10, 10] | 与回归窗分开 |
| 收益 | 简单收益率，AKShare 用前复权 | `adjust` |
| 事件日 | 非交易日向后 10 个日历日内的第一个交易日 | 停牌缺一天则该窗口作废 |
| 聚类 | `firm_id` | 公司是依赖单位 |
| 回归 | pyfixest，公司聚类 | 装不上时退回 statsmodels |
| 星号 | 10% / 5% / 1% | 平均 CAR 的星号来自首选检验，不是横截面 t |
| 单位 | 表和图乘以 100 | CSV 里仍是小数收益率 |
| 安慰剂 | 时间安慰剂与截面安慰剂各 200 次 | 见下 |

首选检验的顺序：事件日有一半以上落在同一天时，用等权组合的时间序列 t；否则若估计窗残差平均相关超过 0.05，用 Kolari–Pynnönen 调整后的 BMP；其余用 BMP。横截面 t 始终列出，但事件日重合或残差相关时会把显著性说得过高。

时间安慰剂把每家公司挪到估计窗之前的其他交易日，问的是「这一天对这批公司是否特别」。截面安慰剂在样本内重配价格序列，问的是「这一天是否特别落到了这些公司上」。两者都不是对照组，p 值也不是处理效应的 p 值。

## 理论备忘

`output/theory/candidate_mechanisms.md` 按事件类型排出三条候选透镜：

- AI 披露先比信号：成本更高、更可核对的披露才应该有更高的 CAR；廉价跟风应接近零。
- ESG 先比合法性：缺口更大时，负向事件的折价应更重；补救不等于自动的正溢价。
- 地缘政治先写现金流和风险。制度压力只解释「一类公司共同面对的新约束」；合法性只解释扣掉经营暴露之后剩下的声誉折价。

`signal_cost`、`legitimacy_gap` 会进入回归。`peer_count_pm5` 是样本内同业事件数，内生，默认不进回归；要那一列时把 `include_peer_count` 设为 true，并在表注里写明。估计做完后按全部系数重写机制，包括不显著的列。本次没有重新检索文献。

## 目录

```
config/study.yml          窗口、聚类、路径
src/ashare_event/         行情、CAR、推断、安慰剂、回归、表、图、理论
app/streamlit_app.py      行业、事件类型、窗口
paper/tabs paper/figs     给论文 \input 和 \includegraphics 的文件
output/theory             估计前的候选机制
output/logs               样本剔除记录
data/raw                  AKShare 原始行情，只写一次
data/processed            事件层 CAR，小数收益率
```

项目放在 `Academic Research/03_Methods_Toolkit`，给各篇论文复用，不单独占一个 `Projects` 文稿夹。
