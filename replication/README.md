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
