# 行情来源

- `mode: demo`：程序内部的市场模型加事件日冲击。不是行情，不能当实证结果。
- `mode: akshare`：`stock_zh_a_hist` 的个股日线，`adjust` 默认前复权；指数用 `index_zh_a_hist`，默认 `000300`。下载时间、区间和 AKShare 版本写在 `data/raw` 同名的 json 里。
- `mode: local`：用户自己的 `code,date,close` 与指数 `date,close`。脚本不改这些文件。

原始文件只在不存在时写入。缓存盖不住新的样本区间时，脚本停止并说明该把旧文件移走，而不是覆盖。

服务器上可以把 `ASHARE_RAW_DIR` 指到持久卷。只有 `ASHARE_EXTEND_CACHE=1` 时，盖不住的区间才会另写一个 `*.extended.parquet`，原来的文件仍然不改。`ASHARE_DOWNLOAD_WORKERS` 控制 AKShare 并发，默认 4。涨跌停标记只按代码板块和公司名称推断带宽，不是交易所状态表。
