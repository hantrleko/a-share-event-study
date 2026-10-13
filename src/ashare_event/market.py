"""Price panels from AKShare, a local file, or an in-memory frame.

Raw downloads are write-once. A later request that falls outside the
cached date range raises, instead of overwriting data/raw.
"""

from __future__ import annotations

import json
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


def fetch_stock(code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
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


def fetch_index(symbol: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    frame = ak.index_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end)
    if frame is None or frame.empty:
        raise MarketDataError(f"AKShare 没有返回指数 {symbol} 的行情。")
    return normalize_ohlc(frame)


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


def load_or_download_stock(
    code: str,
    start,
    end,
    adjust: str,
    raw_dir: Path,
) -> pd.DataFrame:
    path = raw_dir / "prices" / f"{code}.parquet"
    meta_path = path.with_suffix(".json")
    start_ts, end_ts = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        cached_start = pd.Timestamp(meta.get("start", "1900-01-01"))
        cached_end = pd.Timestamp(meta.get("end", "1900-01-01"))
        if cached_start <= start_ts and cached_end >= end_ts and meta.get("adjust") == adjust:
            frame = _read_cache(path)
            frame["date"] = pd.to_datetime(frame["date"])
            return frame
        _coverage_error(path, meta, start_ts.date(), end_ts.date(), adjust)
    frame = fetch_stock(code, _ymd(start_ts), _ymd(end_ts), adjust)
    _write_cache(frame, path)
    meta_path.write_text(
        json.dumps(_meta(start_ts, end_ts, adjust, len(frame), code), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return frame


def load_or_download_index(symbol: str, start, end, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "index" / f"{symbol}.parquet"
    meta_path = path.with_suffix(".json")
    start_ts, end_ts = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        cached_start = pd.Timestamp(meta.get("start", "1900-01-01"))
        cached_end = pd.Timestamp(meta.get("end", "1900-01-01"))
        if cached_start <= start_ts and cached_end >= end_ts:
            frame = _read_cache(path)
            frame["date"] = pd.to_datetime(frame["date"])
            return frame
        _coverage_error(path, meta, start_ts.date(), end_ts.date(), None)
    frame = fetch_index(symbol, _ymd(start_ts), _ymd(end_ts))
    _write_cache(frame, path)
    meta_path.write_text(
        json.dumps(_meta(start_ts, end_ts, None, len(frame), symbol), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return frame


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
