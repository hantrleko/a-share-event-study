"""Command line.

    python -m ashare_event demo
    python -m ashare_event run --config config/study.yml --event-type esg --window -1 1
    python -m ashare_event robustness --config config/study.yml --demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ashare_event.pipeline import run_from_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A股事件研究流水线")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="用模拟样本跑通制表")
    run = sub.add_parser("run", help="按配置文件跑真实或本地样本")
    for item in (demo, run):
        item.add_argument("--config", default="config/study.yml")
        item.add_argument("--industry", action="append", default=None)
        item.add_argument("--event-type", action="append", default=None)
        item.add_argument("--window", nargs=2, type=int, default=None, metavar=("START", "END"))
        item.add_argument("--no-placebo", action="store_true")
    robust = sub.add_parser("robustness", help="另跑模型、缩尾和估计窗对照")
    robust.add_argument("--config", default="config/study.yml")
    robust.add_argument("--demo", action="store_true", help="强制使用模拟样本")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = Path(args.config)
    if args.command == "robustness":
        from ashare_event.config import load_config
        from ashare_event.robustness import run_robustness

        cfg_path = config if config.exists() else Path(__file__).resolve().parents[2] / "config" / "study.yml"
        cfg = load_config(cfg_path)
        if args.demo:
            cfg.mode = "demo"
        frame = run_robustness(cfg)
        dest = cfg.resolve(cfg.paths.get("processed", "data/processed")) / "robustness.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(dest, index=False)
        print(dest)
        return
    if args.command == "demo":
        from ashare_event.config import load_config

        cfg_path = config if config.exists() else Path(__file__).resolve().parents[2] / "config" / "study.yml"
        # demo always forces simulated prices, whatever mode the file happens to be in
        from ashare_event.pipeline import run_study

        cfg = load_config(cfg_path)
        cfg.mode = "demo"
        cfg.verbose = True
        run_study(
            cfg,
            industries=args.industry,
            event_types=args.event_type,
            regression_window=tuple(args.window) if args.window else None,
            run_placebo=not args.no_placebo,
            export=True,
        )
        return
    run_from_file(
        config,
        industries=args.industry,
        event_types=args.event_type,
        regression_window=tuple(args.window) if args.window else None,
        run_placebo=not args.no_placebo,
        export=True,
    )


if __name__ == "__main__":
    main()
