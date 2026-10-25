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

`model` 可以是 `market`、`market_adjusted`、`industry` 或 `ff3`。后两个需要你自己的行业指数或三因子文件，程序不下载。事件表可选列 `after_close`：标成盘后时，从记录日的下一个交易日起对齐。安慰剂在伪日期上重新拟合估计窗。星号来自未校正的首选 p 值；p BH 只调整窗口表里的首选 p 值。

原始行情只写一次。`ASHARE_RAW_DIR` 指向缓存目录；`ASHARE_EXTEND_CACHE=1` 时，盖不住的区间另写 `*.extended.parquet`，不覆盖原文件。
