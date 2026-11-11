# 行情来源

- `mode: demo`：程序内部的市场模型加事件日冲击。不是行情，不能当实证结果。
- `mode: akshare`：`stock_zh_a_hist` 的个股日线，`adjust` 默认前复权；指数用 `index_zh_a_hist`，默认 `000300`。下载时间、区间和 AKShare 版本写在 `data/raw` 同名的 json 里。
- `mode: local`：用户自己的 `code,date,close` 与指数 `date,close`。脚本不改这些文件。

原始文件只在不存在时写入。缓存盖不住新的样本区间时，脚本停止并说明该把旧文件移走，而不是覆盖。

服务器上可以把 `ASHARE_RAW_DIR` 指到持久卷。只有 `ASHARE_EXTEND_CACHE=1` 时，盖不住的区间才会另写一个 `*.extended.parquet`，原来的文件仍然不改。`ASHARE_DOWNLOAD_WORKERS` 控制 AKShare 并发，默认 4。

行业基准不上传文件时，用申万宏源研究的一级行业指数（`index_hist_sw`）和分类历史（`stock_industry_clf_hist_sw`）。分类历史里的代码是行业分类代码，例如 480301。一级行业代码是前两位加 0000，再按申万 2021 公开对照表换成一级指数代码，例如 480000 对 801780。名称以一级指数目录为准。对照表里没有的代码不会猜测。801 开头的指数代码只在和目录完全一致时使用，不用前四位去认父行业。匹配用事件日当天有效的分类，不用事件表里手填的行业名称代替。缓存写在 `data/raw/sw/`，已有文件不覆盖。分类文件若证书校验失败，同一地址会再下载一次。

三因子不上传文件时，读取本机的 Liu-Stambaugh-Yuan CH3 日频表：市场超额收益、SMB、VMG，以及无风险利率。VMG 放进 `hml` 列，样本记录会写明它不是 Fama-French 的 HML。个股收益先减去无风险利率。查找顺序是环境变量 `ASHARE_CH_FACTOR_DIR`，然后是工作区里已下载的 `Projects/TOP2/03_Data_Raw/external/mingshi_china_factors_202511`。这份文件不复制进本仓库。CH3 没有 RMW、CMA、UMD，五因子和 Carhart 仍要上传。

勾选「剔除事件日涨跌停」时，会按日期读取东方财富涨停池、跌停池和停复牌表，并写入 `data/raw/status/`。涨停池不含 ST 和科创板，跌停池只覆盖近 30 日。名单没有覆盖到的，仍用板块带宽。停牌只记录，不额外删行。默认不剔除。

年报词频不在本仓库里下载，也不复制年报文本特征引擎（listed-filing-textmine）的面板或正文。页面按 `ASHARE_FILING_TEXT_DIR`，以及该项目的 `data/factor_panel/output`、`data/workspace/output`、`data/output`、`data/validation/output` 就地读取 `panel_factor`。变量已设置时不再改看其他目录。找不到时可以改传同一份导出。因子表只有 `available_date` 时，事件日用这一天且不再后移，因为该引擎的可用日已经是公告日的次日。同时有公告日、并且可用日是公告日的下一天时，事件日记公告日，`after_close` 为真。`fiscal_year` 只用于收窄报告期，不是事件日。`signal_cost` 在有 `mda_digital_per_k` 时只取这一列。`mda_esg_per_k` 与 `mda_geopol_per_k` 保留原列，不改名为合法性缺口。三列都是管理层讨论的每千词，缺测留空，短文本或核对后缺失的章节也不是强度 0。它们不是信号成本、ESG 表现或地缘冲击的测量。后台任务写在 `data/jobs/`，只在运行时出现，不提交。
