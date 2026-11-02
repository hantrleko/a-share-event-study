# 复现

在项目根目录：

```powershell
$env:PYTHONPATH = "src"
python -m ashare_event demo
```

演示样本的种子是 `config/study.yml` 里的 `seed`。真实样本把 `mode` 改成 `local` 或 `akshare`，并换上自己的事件表后再执行：

```powershell
python -m ashare_event run --config config/study.yml
```

表和图只从这次运行写出。不要手改 `paper/tabs` 里的数字。

命令行不传输出目录时，文件进 `paper/`、`output/` 和 `data/processed/`。网站每次运行使用单独的临时目录，避免同时使用的人互相覆盖。同一次运行还会写出 `manifest.json`、中英文方法段、VIF 和变量字典。

`model` 可以是 `market`、`market_adjusted`、`industry`、`ff3`、`ff5` 或 `carhart`。行业指数和因子文件都要自己提供，程序不下载。五因子需要 `rmw`、`cma`，Carhart 需要 `umd`。事件表可选列 `after_close`：标成盘后时，从记录日的下一个交易日起对齐。安慰剂在伪日期上重新拟合估计窗。星号来自未校正的首选 p 值；p BH 只调整窗口表里的首选 p 值。

对照规格不进主程序。需要时：

```powershell
python -m ashare_event robustness --demo
```

每次导出的 `logs/replication/` 里有 `study_repro.yml`、`events.csv`、`run_replication.py` 和 `replicate_regressions.do`。演示模式按种子重放。这份目录不含行情。`reghdfe` 需自行安装，`.do` 里的系数是相关，不是因果。

原始行情只写一次。`ASHARE_RAW_DIR` 指向缓存目录；`ASHARE_EXTEND_CACHE=1` 时，盖不住的区间另写 `*.extended.parquet`，不覆盖原文件。
