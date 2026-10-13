"""Command line.

    python -m ashare_event demo
    python -m ashare_event run --config config/study.yml --event-type esg --window -1 1
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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = Path(args.config)
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
