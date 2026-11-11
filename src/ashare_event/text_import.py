"""Turn a filing-text export into an event table.

The text project is not vendored. This module only reads a table the user
already exported (factor panel, wide panel, long panel, or regression readiness).

Date rule, matching that project's pit convention:
- available_date is announcement T+1 (a calendar day, not a trading day).
- If announce_date or filing_date exists and available_date is the next calendar
  day, the event date is the announcement and after_close is true.
- If the file only has available_date, the event date is that day and after_close
  stays false, so the study does not shift T+1 a second time.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ashare_event.schema import SchemaError, normalize_code

_CODE_ALIASES = {
    "stkcd": "stkcd",
    "ticker": "ticker",
    "code": "code",
    "证券代码": "stkcd",
    "股票代码": "stkcd",
}
_DATE_ALIASES = {
    "announce_date": "announce_date",
    "announcement_date": "announce_date",
    "filing_date": "filing_date",
    "available_date": "available_date",
    "公告日期": "announce_date",
    "公告日": "announce_date",
    "可用日": "available_date",
    "披露日期": "announce_date",
}
_OTHER_ALIASES = {
    "firm_name": "firm_name",
    "公司名称": "firm_name",
    "market": "market",
    "section": "section",
    "mda_digital_per_k": "mda_digital_per_k",
    "mda_esg_per_k": "mda_esg_per_k",
    "mda_geopol_per_k": "mda_geopol_per_k",
    "digital_per_k": "digital_per_k",
    "esg_per_k": "esg_per_k",
    "geopol_per_k": "geopol_per_k",
    "pit_ready": "pit_ready",
    "report_type": "report_type",
    "industry": "industry",
}
_A_SHARE_MARKETS = {"", "CN", "A", "ASHARE", "A股", "SH", "SZ", "BJ", "SSE", "SZSE", "SHSE", "CHINA"}
_MDA_SECTIONS = {"mda", "md&a", "管理层讨论", "管理层讨论与分析"}
# Word frequencies from the annual-report text engine (listed-filing-textmine).
# signal_cost is only the MD&A digital/AI intensity. ESG and geopolitics stay
# in their own columns and are not renamed to legitimacy_gap.
_THEMES = (
    ("_signal", "mda_digital_per_k", "digital_per_k", "signal_cost"),
    ("_esg", "mda_esg_per_k", "esg_per_k", "mda_esg_per_k"),
    ("_geopol", "mda_geopol_per_k", "geopol_per_k", "mda_geopol_per_k"),
)


def read_filing_table(source, name: str = "") -> pd.DataFrame:
    """Read csv, parquet, or dta. Does not open the text project's own folders."""
    label = name or getattr(source, "name", "") or (str(source) if isinstance(source, (str, Path)) else "")
    suffix = Path(label).suffix.lower()
    try:
        if hasattr(source, "read") and not isinstance(source, (str, Path)):
            payload = source.read()
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            buffer = io.BytesIO(payload)
            frame = _read_buffer(buffer, suffix)
        else:
            path = Path(source)
            suffix = path.suffix.lower()
            label = path.name
            frame = _read_path(path, suffix)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"这份年报文本面板读不出来（{label or '未命名文件'}）。请用 csv、parquet 或 dta。") from exc
    if frame is None or not len(frame.columns):
        raise ValueError("年报文本面板是空的。")
    return frame


def events_from_filing_text(frame: pd.DataFrame) -> pd.DataFrame:
    """Map one exported panel onto code, event_date, after_close, and signal_cost."""
    if frame is None or len(frame) == 0:
        raise ValueError("年报文本面板是空的。")
    data = _canonicalize(frame)
    if not any(column in data.columns for column in ("stkcd", "ticker", "code")):
        raise ValueError("年报文本面板里没有证券代码。需要 stkcd、ticker 或 code。")
    if not any(column in data.columns for column in ("announce_date", "filing_date", "available_date")):
        raise ValueError("年报文本面板里没有公告日或可用日。")

    n_in = len(data)
    kept: list[dict] = []
    n_market = 0
    n_bad = 0
    n_shift = 0
    n_available = 0
    present = {
        internal: wide in data.columns or long in data.columns
        for internal, wide, long, _output in _THEMES
    }
    for record in data.to_dict(orient="records"):
        if not _is_a_share(record.get("market")):
            n_market += 1
            continue
        code = _code_of(record)
        event_date, after_close = _event_day(record)
        if code is None or event_date is None:
            n_bad += 1
            continue
        if after_close:
            n_shift += 1
        else:
            n_available += 1
        kept.append(
            {
                "code": code,
                "event_date": event_date,
                "event_type": _event_type(record.get("report_type")),
                "industry": _text(record.get("industry")) or "unspecified",
                "firm_name": _text(record.get("firm_name")),
                "after_close": bool(after_close),
                "_signal": _theme(record, "mda_digital_per_k", "digital_per_k"),
                "_esg": _theme(record, "mda_esg_per_k", "esg_per_k"),
                "_geopol": _theme(record, "mda_geopol_per_k", "geopol_per_k"),
                "fiscal_year": _year(record.get("fiscal_year")) if "fiscal_year" in data.columns else None,
                "_pit": _pit(record.get("pit_ready")),
                "_order": len(kept),
            }
        )
    if not kept:
        raise ValueError("这份表里没有可用的 A 股公告日。市场列不是 CN 的行已经跳过。")

    work = pd.DataFrame(kept)
    work = work.sort_values(["_pit", "_order"], ascending=[False, True])
    identity = work.drop_duplicates(["code", "event_date"], keep="first")
    n_collapsed = len(work) - len(identity)
    out = identity.drop(columns=["_pit", "_order"])
    for internal, _wide, _long, output in _THEMES:
        filled = work.dropna(subset=[internal]).drop_duplicates(["code", "event_date"], keep="first")
        out = out.drop(columns=[internal]).merge(
            filled.loc[:, ["code", "event_date", internal]],
            on=["code", "event_date"],
            how="left",
        )
        out = out.rename(columns={internal: output})
        if not present[internal]:
            out = out.drop(columns=[output])
        elif out[output].isna().all():
            out[output] = np.nan
    out["event_date"] = pd.to_datetime(out["event_date"])
    if "fiscal_year" in out.columns and out["fiscal_year"].isna().all():
        out = out.drop(columns=["fiscal_year"])
    out = out.sort_values(["event_date", "code"]).reset_index(drop=True)
    has_frequency = present["_signal"]

    note = _note(
        n_in=n_in,
        n_out=len(out),
        n_market=n_market,
        n_bad=n_bad,
        n_collapsed=n_collapsed,
        n_shift=n_shift,
        n_available=n_available,
        has_frequency=has_frequency,
        n_missing=int(out["signal_cost"].isna().sum()) if "signal_cost" in out.columns else 0,
    )
    out.attrs["import_note"] = note
    out.attrs["signal_source"] = "mda_digital_per_k" if has_frequency else ""
    return out


def _read_buffer(buffer: io.BytesIO, suffix: str) -> pd.DataFrame:
    if suffix == ".parquet":
        return pd.read_parquet(buffer)
    if suffix == ".dta":
        return pd.read_stata(buffer)
    if suffix not in {"", ".csv", ".txt"}:
        raise ValueError("年报文本面板只接受 csv、parquet 或 dta。")
    return pd.read_csv(buffer)


def _read_path(path: Path, suffix: str) -> pd.DataFrame:
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".dta":
        return pd.read_stata(path)
    if suffix not in {"", ".csv", ".txt"}:
        raise ValueError("年报文本面板只接受 csv、parquet 或 dta。")
    return pd.read_csv(path)


def _canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {**_CODE_ALIASES, **_DATE_ALIASES, **_OTHER_ALIASES}
    rename = {}
    for column in frame.columns:
        key = str(column).strip().lstrip("\ufeff")
        target = aliases.get(key) or aliases.get(key.lower())
        if target:
            rename[column] = target
    data = frame.rename(columns=rename)
    return data.loc[:, ~data.columns.duplicated()].copy()


def _is_a_share(value) -> bool:
    if _missing(value):
        return True
    return str(value).strip().upper() in _A_SHARE_MARKETS


def _code_of(record: dict) -> str | None:
    raw = record.get("stkcd")
    if _missing(raw):
        raw = record.get("ticker")
    if _missing(raw):
        raw = record.get("code")
    if _missing(raw):
        return None
    try:
        return normalize_code(raw)
    except SchemaError:
        return None


def _event_day(record: dict) -> tuple[pd.Timestamp | None, bool]:
    announce = _day(record.get("announce_date")) or _day(record.get("filing_date"))
    available = _day(record.get("available_date"))
    if announce is not None and available is not None:
        if available == announce + pd.Timedelta(days=1):
            return announce, True
        return available, False
    if announce is not None:
        return announce, True
    if available is not None:
        return available, False
    return None, False


def _theme(record: dict, wide_name: str, long_name: str) -> float | None:
    """MD&A intensity from the annual-report text engine. A risk-section row does not supply it."""
    if wide_name in record:
        value = _finite(record.get(wide_name))
        if value is not None or "section" not in record:
            return value
    section = _text(record.get("section")).lower()
    if section and section not in _MDA_SECTIONS and "mda" not in section:
        return None
    if long_name in record:
        return _finite(record.get(long_name))
    return None


def _event_type(value) -> str:
    text = _text(value)
    return text or "annual_report"


def _year(value) -> int | None:
    number = _finite(value)
    if number is None:
        return None
    year = int(number)
    if year < 1900 or year > 2100:
        return None
    return year


def _pit(value) -> int:
    number = _finite(value)
    if number is None:
        return 0
    return 1 if number > 0 else 0


def _day(value) -> pd.Timestamp | None:
    if _missing(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def _finite(value) -> float | None:
    if _missing(value):
        return None
    number = pd.to_numeric(value, errors="coerce")
    try:
        if pd.isna(number):
            return None
    except TypeError:
        return None
    number = float(number)
    if not np.isfinite(number):
        return None
    return number


def _text(value) -> str:
    if _missing(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat", "<na>"}:
        return ""
    return text


def _missing(value) -> bool:
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        return False
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _note(
    *,
    n_in: int,
    n_out: int,
    n_market: int,
    n_bad: int,
    n_collapsed: int,
    n_shift: int,
    n_available: int,
    has_frequency: bool,
    n_missing: int,
) -> str:
    bits = [f"读入 {n_in} 行，留下 {n_out} 条公司-日期。"]
    if n_market:
        bits.append(f"跳过非 A 股 {n_market} 行。")
    if n_bad:
        bits.append(f"没有代码或日期的 {n_bad} 行已跳过。")
    if n_collapsed:
        bits.append(f"同一公司同一天合并掉 {n_collapsed} 行，保留 pit_ready 优先的一行，词频不取平均。")
    if n_shift and n_available:
        bits.append("有公告日且可用日正好是下一天的，事件日记公告日并勾上盘后。")
        bits.append("只有可用日、或可用日不是下一天的，事件日用可用日，不再后移。")
    elif n_shift:
        bits.append("事件日记公告日或申报日，并勾上盘后。年报文本特征引擎的可用日是公告日的次日，这里不再多移一个交易日。")
    elif n_available:
        bits.append("事件日用可用日，盘后不勾。可用日在年报文本特征引擎里已经是公告日的次日。")
    if has_frequency:
        bits.append(
            "词频来自年报文本特征引擎的管理层讨论。signal_cost 只取数智化/AI 每千词，当作披露具体程度的代理，不是已经测得的信号成本。"
            "绿色/ESG 与地缘每千词各自留在原列，不写入合法性缺口。"
        )
        if n_missing:
            bits.append(f"其中 {n_missing} 条没有数智化词频，留空，没有记成 0。")
    return "".join(bits)


_PANEL_NAMES = (
    "panel_factor.parquet",
    "panel_factor.dta",
    "panel_factor.csv",
    "panel_wide.parquet",
    "regression_readiness.csv",
)
_PANEL_DIRS = (
    "",
    "output",
    "data/factor_panel/output",
    "data/workspace/output",
    "data/output",
    "data/validation/output",
)
_KNOWN_FILING_ROOT = Path(r"F:\listed-filing-textmine")


def locate_filing_panel() -> Path | None:
    """Find the annual-report text engine's factor table without copying it.

    ASHARE_FILING_TEXT_DIR may be the file or the project/output directory.
    When that variable is set, no other location is searched.
    """
    override = os.environ.get("ASHARE_FILING_TEXT_DIR", "").strip()
    if override:
        return _panel_in(Path(override))
    return _panel_in(_KNOWN_FILING_ROOT)


def load_filing_events(path: Path) -> pd.DataFrame:
    """Read one engine export and map it onto the event-study table."""
    events = events_from_filing_text(read_filing_table(path))
    events.attrs["filing_path"] = str(path)
    return events


def _panel_in(root: Path) -> Path | None:
    """Prefer the factor table over a wide file that happens to sit in an earlier folder."""
    if root.is_file() and root.name in _PANEL_NAMES:
        return root
    if not root.exists():
        return None
    for name in _PANEL_NAMES:
        for folder in _PANEL_DIRS:
            path = (root if not folder else root / folder) / name
            if path.is_file():
                return path
    return None
