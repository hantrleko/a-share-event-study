"""Load and validate study configuration.

Window rules are enforced here so a CAR is never estimated on days
that were also used to fit the market model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """The study configuration contradicts the event-study design."""


@dataclass
class StudyConfig:
    project_name: str
    seed: int
    mode: str
    language: str
    model: str
    benchmark: str
    adjust: str
    estimation_window: tuple[int, int]
    event_windows: list[tuple[int, int]]
    path_window: tuple[int, int]
    regression_window: tuple[int, int]
    min_estimation_obs: int
    require_complete_window: bool
    cluster: str
    covariates: list[str]
    include_peer_count: bool
    return_scale: float
    placebo_draws: int
    placebo_offsets: tuple[int, int]
    placebo_modes: list[str]
    rbar_min_periods: int
    portfolio_min_share: float
    figure_size: tuple[float, float]
    events_file: str | None
    prices_file: str | None
    market_file: str | None
    paths: dict[str, str]
    root: Path
    verbose: bool = False
    extra: dict = field(default_factory=dict)

    def resolve(self, rel: str) -> Path:
        path = Path(rel)
        if path.is_absolute():
            return path
        return (self.root / path).resolve()


def _pair(value, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{name} 需要两个整数 [起点, 终点]。")
    left, right = int(value[0]), int(value[1])
    if left > right:
        raise ConfigError(f"{name} 的起点晚于终点：{left} > {right}。")
    return left, right


def _windows(value) -> list[tuple[int, int]]:
    if not value:
        raise ConfigError("event_windows 不能为空。")
    return [_pair(item, "event_windows") for item in value]


def validate_windows(cfg: StudyConfig) -> None:
    """Estimation days must end before every reported event window."""
    est_end = cfg.estimation_window[1]
    blockers = [("path_window", cfg.path_window), *[(f"event {w}", w) for w in cfg.event_windows]]
    if cfg.regression_window not in cfg.event_windows:
        blockers.append(("regression_window", cfg.regression_window))
    for name, (left, _right) in blockers:
        if left <= est_end:
            raise ConfigError(
                f"{name} 从 {left} 开始，估计窗结束于 {est_end}。"
                "两者重叠时，事件日会进入市场模型，异常收益被压向 0。"
                "请把估计窗的右端点放到所有事件窗和路径窗的左边。"
            )
    if cfg.model not in {"market", "market_adjusted"}:
        raise ConfigError("model 只能是 market 或 market_adjusted。")
    if cfg.mode not in {"demo", "local", "akshare"}:
        raise ConfigError("mode 只能是 demo、local 或 akshare。")
    if cfg.language not in {"zh", "en"}:
        raise ConfigError("language 只能是 zh 或 en。")
    low, high = cfg.placebo_offsets
    if high < 0 and high >= cfg.estimation_window[1]:
        # Negative shifts that land inside the estimation window are allowed
        # as pseudo-event dates; the shock contamination check is separate.
        pass
    if low > high:
        raise ConfigError("placebo_offsets 的左端不能大于右端。")
    for mode in cfg.placebo_modes:
        if mode not in {"in_time", "in_space"}:
            raise ConfigError(f"未知安慰剂类型：{mode}")
    if cfg.min_estimation_obs < 30:
        raise ConfigError("min_estimation_obs 至少为 30，否则市场模型标准误没有意义。")
    if cfg.return_scale <= 0:
        raise ConfigError("return_scale 必须为正。")


def load_config(path: str | Path, **overrides) -> StudyConfig:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    root = path.parent.parent if path.parent.name == "config" else path.parent
    cfg = StudyConfig(
        project_name=str(raw.get("project_name", "a-share-event-study")),
        seed=int(raw.get("seed", 20240101)),
        mode=str(raw.get("mode", "demo")),
        language=str(raw.get("language", "zh")),
        model=str(raw.get("model", "market")),
        benchmark=str(raw.get("benchmark", "000300")),
        adjust=str(raw.get("adjust", "qfq")),
        estimation_window=_pair(raw.get("estimation_window", [-250, -31]), "estimation_window"),
        event_windows=_windows(raw.get("event_windows", [[-1, 1], [-5, 5]])),
        path_window=_pair(raw.get("path_window", [-10, 10]), "path_window"),
        regression_window=_pair(raw.get("regression_window", [-1, 1]), "regression_window"),
        min_estimation_obs=int(raw.get("min_estimation_obs", 80)),
        require_complete_window=bool(raw.get("require_complete_window", True)),
        cluster=str(raw.get("cluster", "firm_id")),
        covariates=list(raw.get("covariates", ["signal_cost", "size_log", "legitimacy_gap"])),
        include_peer_count=bool(raw.get("include_peer_count", False)),
        return_scale=float(raw.get("return_scale", 100)),
        placebo_draws=int(raw.get("placebo_draws", 200)),
        placebo_offsets=_pair(raw.get("placebo_offsets", [-180, -40]), "placebo_offsets"),
        placebo_modes=list(raw.get("placebo_modes", ["in_time"])),
        rbar_min_periods=int(raw.get("rbar_min_periods", 30)),
        portfolio_min_share=float(raw.get("portfolio_min_share", 0.5)),
        figure_size=tuple(float(x) for x in raw.get("figure_size", [6.5, 4.0])),
        events_file=raw.get("events_file"),
        prices_file=raw.get("prices_file"),
        market_file=raw.get("market_file"),
        paths=dict(raw.get("paths", {})),
        root=root,
        extra={k: v for k, v in raw.items() if k not in _KNOWN},
    )
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise ConfigError(f"未知配置项：{key}")
        setattr(cfg, key, value)
    validate_windows(cfg)
    return cfg


_KNOWN = {
    "project_name",
    "seed",
    "mode",
    "language",
    "model",
    "benchmark",
    "adjust",
    "estimation_window",
    "event_windows",
    "path_window",
    "regression_window",
    "min_estimation_obs",
    "require_complete_window",
    "cluster",
    "covariates",
    "include_peer_count",
    "return_scale",
    "placebo_draws",
    "placebo_offsets",
    "placebo_modes",
    "rbar_min_periods",
    "portfolio_min_share",
    "figure_size",
    "events_file",
    "prices_file",
    "market_file",
    "paths",
}


def window_suffix(left: int, right: int) -> str:
    def part(value: int) -> str:
        return f"m{abs(value)}" if value < 0 else f"p{value}"

    return f"{part(left)}_{part(right)}"


def window_label(left: int, right: int) -> str:
    return f"[{left}, {right}]"
