"""Price panels from AKShare, a local file, or an in-memory frame.

Raw downloads are write-once. A later request that falls outside the
cached date range raises, instead of overwriting data/raw.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATE_NAMES = {"日期", "date", "交易日", "time"}
CLOSE_NAMES = {"收盘", "close", "收盘价"}


class MarketDataError(RuntimeError):
    """Prices could not be loaded or the raw cache does not cover the sample."""


def normalize_ohlc(frame: pd.DataFrame, code: str | None = None) -> pd.DataFrame:
    rename = {}
    for column in frame.columns:
        key = str(column).strip()
        if key.lower() in DATE_NAMES or key in DATE_NAMES:
            rename[column] = "date"
        elif key.lower() in CLOSE_NAMES or key in CLOSE_NAMES:
            rename[column] = "close"
    out = frame.rename(columns=rename)
    if "date" not in out.columns or "close" not in out.columns:
        raise MarketDataError(f"无法识别行情列：{list(frame.columns)}")
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"])
    out = out.loc[out["close"] > 0].copy()
    if code is not None:
        out["code"] = code
    keep = ["date", "close"] if code is None else ["code", "date", "close"]
    return out[keep].drop_duplicates(subset=keep[:2] if code is None else ["code", "date"])


def _ymd(value) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _retry(action, label: str):
    """Retry transient download errors. An explicit empty response is not retried."""
    last = None
    for attempt in range(3):
        try:
            return action()
        except MarketDataError:
            raise
        except Exception as exc:
            last = exc
            if attempt == 2:
                break
            time.sleep(0.8 * (attempt + 1))
    raise MarketDataError(f"{label} 连续 3 次没有成功：{last}")


def fetch_stock(code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    def once():
        import akshare as ak

        frame = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust=adjust,
        )
        if frame is None or frame.empty:
            raise MarketDataError(f"AKShare 没有返回 {code} 的行情。")
        return normalize_ohlc(frame, code=code)

    return _retry(once, f"{code} 行情")


def fetch_index(symbol: str, start: str, end: str) -> pd.DataFrame:
    def once():
        import akshare as ak

        frame = ak.index_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end)
        if frame is None or frame.empty:
            raise MarketDataError(f"AKShare 没有返回指数 {symbol} 的行情。")
        return normalize_ohlc(frame)

    return _retry(once, f"指数 {symbol}")


def resolve_raw_dir(root: Path) -> Path:
    """Raw cache directory. ASHARE_RAW_DIR can point at a persistent volume."""
    override = os.environ.get("ASHARE_RAW_DIR", "").strip()
    if override:
        return Path(override)
    return Path(root) / "data" / "raw"


def _extend_enabled() -> bool:
    return os.environ.get("ASHARE_EXTEND_CACHE", "").strip().lower() in {"1", "true", "yes"}


def _find_column(frame: pd.DataFrame, names: set[str]):
    wanted = {name.lower() for name in names}
    for column in frame.columns:
        key = str(column).strip()
        if key in names or key.lower() in wanted:
            return column
    return None


def _meta_covers(meta: dict, start_ts, end_ts, adjust: str | None) -> bool:
    cached_start = pd.Timestamp(meta.get("start", "1900-01-01"))
    cached_end = pd.Timestamp(meta.get("end", "1900-01-01"))
    if not (cached_start <= start_ts and cached_end >= end_ts):
        return False
    if adjust is None:
        return True
    return meta.get("adjust") == adjust


def _load_if_covers(path: Path, start_ts, end_ts, adjust: str | None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if not _meta_covers(meta, start_ts, end_ts, adjust):
        return None
    frame = _read_cache(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _read_cache(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["date"])


def _write_cache(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise MarketDataError(f"原始行情已存在，拒绝覆盖：{path}")
    frame.to_parquet(path, index=False)


def _coverage_error(path: Path, have: dict, start, end, adjust: str | None) -> None:
    raise MarketDataError(
        "原始缓存的区间或复权方式盖不住这次样本，脚本不会改写 data/raw。"
        f"文件：{path}。已有 {have}。本次需要 {start} 至 {end}，adjust={adjust}。"
        "请把旧文件移出 data/raw 后再下载。"
    )


def _read_meta(path: Path) -> dict:
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _store_download(frame: pd.DataFrame, path: Path, start_ts, end_ts, adjust, symbol: str) -> None:
    """Write a new cache file. An existing file is never replaced."""
    _write_cache(frame, path)
    path.with_suffix(".json").write_text(
        json.dumps(_meta(start_ts, end_ts, adjust, len(frame), symbol), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_or_download_stock(
    code: str,
    start,
    end,
    adjust: str,
    raw_dir: Path,
) -> pd.DataFrame:
    path = raw_dir / "prices" / f"{code}.parquet"
    start_ts, end_ts = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    loaded = _load_if_covers(path, start_ts, end_ts, adjust)
    if loaded is not None:
        return loaded
    if path.exists() and not _extend_enabled():
        _coverage_error(path, _read_meta(path), start_ts.date(), end_ts.date(), adjust)
    if path.exists():
        side = path.with_name(path.stem + ".extended.parquet")
        loaded = _load_if_covers(side, start_ts, end_ts, adjust)
        if loaded is not None:
            return loaded
        if side.exists():
            _coverage_error(side, _read_meta(side), start_ts.date(), end_ts.date(), adjust)
        frame = fetch_stock(code, _ymd(start_ts), _ymd(end_ts), adjust)
        _store_download(frame, side, start_ts, end_ts, adjust, code)
        return frame
    frame = fetch_stock(code, _ymd(start_ts), _ymd(end_ts), adjust)
    _store_download(frame, path, start_ts, end_ts, adjust, code)
    return frame


def load_or_download_index(symbol: str, start, end, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "index" / f"{symbol}.parquet"
    start_ts, end_ts = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    loaded = _load_if_covers(path, start_ts, end_ts, None)
    if loaded is not None:
        return loaded
    if path.exists() and not _extend_enabled():
        _coverage_error(path, _read_meta(path), start_ts.date(), end_ts.date(), None)
    if path.exists():
        side = path.with_name(path.stem + ".extended.parquet")
        loaded = _load_if_covers(side, start_ts, end_ts, None)
        if loaded is not None:
            return loaded
        if side.exists():
            _coverage_error(side, _read_meta(side), start_ts.date(), end_ts.date(), None)
        frame = fetch_index(symbol, _ymd(start_ts), _ymd(end_ts))
        _store_download(frame, side, start_ts, end_ts, None, symbol)
        return frame
    frame = fetch_index(symbol, _ymd(start_ts), _ymd(end_ts))
    _store_download(frame, path, start_ts, end_ts, None, symbol)
    return frame


def read_industry_frame(frame: pd.DataFrame) -> pd.DataFrame:
    industry = _find_column(frame, {"industry", "行业", "所属行业", "申万行业"})
    dated = _find_column(frame, DATE_NAMES)
    close = _find_column(frame, CLOSE_NAMES)
    if industry is None or dated is None or close is None:
        raise MarketDataError("行业指数需要行业、日期、收盘三列。")
    out = pd.DataFrame(
        {
            "industry": frame[industry].astype(str).str.strip(),
            "date": pd.to_datetime(frame[dated], errors="coerce").dt.normalize(),
            "close": pd.to_numeric(frame[close], errors="coerce"),
        }
    ).dropna()
    out = out.loc[out["close"] > 0]
    if out.empty:
        raise MarketDataError("行业指数里没有可用的收盘价。")
    return out.reset_index(drop=True)


def prepare_industry_returns(frame: pd.DataFrame | None) -> dict[str, pd.Series]:
    if frame is None or len(frame) == 0:
        return {}
    clean = frame if {"industry", "date", "close"}.issubset(frame.columns) else read_industry_frame(frame)
    out: dict[str, pd.Series] = {}
    for name, part in clean.groupby(clean["industry"].astype(str), sort=False):
        ordered = part.sort_values("date").drop_duplicates("date")
        out[str(name)] = ordered.set_index("date")["close"].pct_change()
    return out


def read_factor_frame(frame: pd.DataFrame) -> pd.DataFrame:
    dated = _find_column(frame, DATE_NAMES | {"日期"})
    mkt = _find_column(frame, {"mkt", "mkt_rf", "市场", "市场因子"})
    smb = _find_column(frame, {"smb", "规模", "规模因子"})
    hml = _find_column(frame, {"hml", "价值", "价值因子"})
    if dated is None or mkt is None or smb is None or hml is None:
        raise MarketDataError("三因子文件需要日期、mkt、smb、hml。")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[dated], errors="coerce").dt.normalize(),
            "mkt": pd.to_numeric(frame[mkt], errors="coerce"),
            "smb": pd.to_numeric(frame[smb], errors="coerce"),
            "hml": pd.to_numeric(frame[hml], errors="coerce"),
        }
    ).dropna()
    if out.empty:
        raise MarketDataError("三因子文件里没有同时有日期和三个因子的行。")
    return out.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def prepare_factors(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None or len(frame) == 0:
        return None
    if {"date", "mkt", "smb", "hml"}.issubset(frame.columns):
        out = frame[["date", "mkt", "smb", "hml"]].copy()
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        for column in ("mkt", "smb", "hml"):
            out[column] = pd.to_numeric(out[column], errors="coerce")
        return out.dropna().drop_duplicates("date")
    return read_factor_frame(frame)


def _meta(start, end, adjust, rows: int, symbol: str) -> dict:
    version = None
    try:
        import akshare as ak

        version = getattr(ak, "__version__", None)
    except Exception:
        version = None
    return {
        "symbol": symbol,
        "start": str(pd.Timestamp(start).date()),
        "end": str(pd.Timestamp(end).date()),
        "adjust": adjust,
        "rows": rows,
        "akshare": version,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }


def read_price_frame(frame: pd.DataFrame, code: str | None = None) -> pd.DataFrame:
    code_column = next((column for column in frame.columns if str(column).strip() in {"code", "股票代码", "证券代码"}), None)
    if code_column is not None:
        labelled = frame.rename(columns={code_column: "code"})
        pieces = []
        for firm, part in labelled.groupby(labelled["code"].map(lambda value: str(value).zfill(6)[-6:]), sort=False):
            pieces.append(normalize_ohlc(part, code=str(firm)))
        if not pieces:
            raise MarketDataError("行情文件里没有可用的收盘价。")
        return pd.concat(pieces, ignore_index=True)
    return normalize_ohlc(frame, code=code)


def read_price_file(path: Path, code: str | None = None) -> pd.DataFrame:
    if not path.exists():
        raise MarketDataError(f"找不到行情文件：{path}")
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    return read_price_frame(frame, code=code)


def sample_date_range(events: pd.DataFrame, estimation_left: int, placebo_left: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Calendar buffer large enough for the estimation window and in-time placebos."""
    span = int(abs(estimation_left) * 2 + abs(min(placebo_left, 0)) * 2 + 45)
    start = pd.Timestamp(events["event_date"].min()) - pd.Timedelta(days=span)
    end = pd.Timestamp(events["event_date"].max()) + pd.Timedelta(days=45)
    return start.normalize(), end.normalize()
