"""Built-in Shenwan industries, CH3 factors, and exchange trading-status lists.

Prices in data/raw stay write-once. These helpers add sibling caches under
data/raw/sw and data/raw/status. They do not overwrite an existing file.
The CH3 file is the Liu-Stambaugh-Yuan daily series already on this machine.
It is not a Fama-French five-factor file: there is no RMW, CMA, or UMD.
"""

from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path

import pandas as pd

from ashare_event.market import MarketDataError, resolve_raw_dir

CASE_ID = "ai_policy_20230713"

# Shenwan 2021 level-1 classification code -> published index code.
# A membership row stores a level-2 or level-3 code such as 480301.
# Its level-1 classification code is the first two digits plus 0000.
# The name used in estimation comes from the level-1 index catalog.
# Codes outside these 31 parents are left unmatched.
SW2021_L1_CLASS_TO_INDEX = {
    "110000": "801010",
    "220000": "801030",
    "230000": "801040",
    "240000": "801050",
    "270000": "801080",
    "280000": "801880",
    "330000": "801110",
    "340000": "801120",
    "350000": "801130",
    "360000": "801140",
    "370000": "801150",
    "410000": "801160",
    "420000": "801170",
    "430000": "801180",
    "450000": "801200",
    "460000": "801210",
    "480000": "801780",
    "490000": "801790",
    "510000": "801230",
    "610000": "801710",
    "620000": "801720",
    "630000": "801730",
    "640000": "801890",
    "650000": "801740",
    "710000": "801750",
    "720000": "801760",
    "730000": "801770",
    "740000": "801950",
    "750000": "801960",
    "760000": "801970",
    "770000": "801980",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def case_events_path() -> Path:
    return repo_root() / "data" / "cases" / f"{CASE_ID}.csv"


def case_summary_path() -> Path:
    return repo_root() / "data" / "cases" / f"{CASE_ID}.json"


def load_case_events() -> pd.DataFrame:
    from ashare_event.schema import load_events

    path = case_events_path()
    if not path.exists():
        raise FileNotFoundError(path)
    return load_events(pd.read_csv(path))


def load_case_summary() -> dict | None:
    path = case_summary_path()
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mean_car_pp") is None or payload.get("n") is None:
        return None
    return payload


def _academic_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "Projects" / "TOP2").exists():
            return parent
    return None


def ch3_candidates() -> list[Path]:
    found: list[Path] = []
    override = os.environ.get("ASHARE_CH_FACTOR_DIR", "").strip()
    if override:
        found.append(Path(override) / "CH3_factors_daily_202511.xlsx")
    academic = _academic_root()
    if academic is not None:
        found.append(
            academic
            / "Projects"
            / "TOP2"
            / "03_Data_Raw"
            / "external"
            / "mingshi_china_factors_202511"
            / "CH3_factors_daily_202511.xlsx"
        )
    return found


def locate_ch3() -> Path | None:
    for path in ch3_candidates():
        if path.exists():
            return path
    return None


def _parse_factor_date(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        text = series.round().astype("Int64").astype(str)
        return pd.to_datetime(text, format="%Y%m%d", errors="coerce").dt.normalize()
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def ch_frame_from_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Map a CH3 sheet onto date, mkt, smb, hml, rf.

    VMG is stored in the hml column so the existing three-factor estimator can
    read it. The sample log says it is VMG, not Fama-French HML.
    """
    columns = {str(column).strip().lower(): column for column in frame.columns}
    date_key = columns.get("date") or columns.get("trade_date")
    mkt_key = columns.get("mktrf") or columns.get("mkt")
    smb_key = columns.get("smb")
    value_key = columns.get("vmg") or columns.get("hml")
    rf_key = columns.get("rf_dly") or columns.get("rf")
    if date_key is None or mkt_key is None or smb_key is None or value_key is None or rf_key is None:
        raise MarketDataError("CH3 日频表需要 date、rf_dly、mktrf、SMB、VMG。")
    out = pd.DataFrame(
        {
            "date": _parse_factor_date(frame[date_key]),
            "mkt": pd.to_numeric(frame[mkt_key], errors="coerce"),
            "smb": pd.to_numeric(frame[smb_key], errors="coerce"),
            "hml": pd.to_numeric(frame[value_key], errors="coerce"),
            "rf": pd.to_numeric(frame[rf_key], errors="coerce"),
        }
    ).dropna()
    if out.empty:
        raise MarketDataError("CH3 日频表里没有同时有日期和四个收益列的行。")
    scale = float(out["mkt"].abs().median())
    if scale > 0.2:
        for name in ("mkt", "smb", "hml", "rf"):
            out[name] = out[name] / 100.0
    out["factor_source"] = "ch3_vmg"
    return out.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def load_ch3_factors(path: Path | None = None) -> pd.DataFrame:
    chosen = path or locate_ch3()
    if chosen is None:
        looked = "；".join(str(item) for item in ch3_candidates()) or "未配置 ASHARE_CH_FACTOR_DIR"
        raise MarketDataError(
            "没有找到 Liu-Stambaugh-Yuan CH3 日频文件。这份公开库提供市场超额收益、SMB 和 VMG，"
            f"没有 RMW、CMA、UMD。已查找：{looked}"
        )
    sheet = pd.read_excel(chosen, sheet_name="Returnseries")
    frame = ch_frame_from_table(sheet)
    frame.attrs["path"] = str(chosen)
    return frame


def normalize_sw_code(value) -> str:
    text = str(value).strip().upper().replace(".SI", "")
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    digits = "".join(character for character in text if character.isdigit())
    return digits[-6:] if len(digits) >= 6 else digits


def class_l1_index(industry_code: str) -> str | None:
    """Map a Shenwan classification code to its 2021 level-1 index code.

    480301 becomes 480000, then 801780. Index codes that start with 8 are
    not rewritten: 801010 and 801020 stay distinct, and 850111 is not a parent.
    """
    code = normalize_sw_code(industry_code)
    if len(code) != 6 or not code.isdigit() or code.startswith("8"):
        return None
    return SW2021_L1_CLASS_TO_INDEX.get(code[:2] + "0000")


def _stock_code(value) -> str:
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    digits = "".join(character for character in text if character.isdigit())
    return digits.zfill(6)[-6:] if digits else ""


def normalize_membership(frame: pd.DataFrame) -> pd.DataFrame:
    """Return code, start, industry_code, industry_name from a Shenwan history file."""
    columns = {str(column).strip(): column for column in frame.columns}
    lowered = {key.lower(): column for key, column in columns.items()}

    def pick(*names: str):
        for name in names:
            if name in columns:
                return columns[name]
            if name.lower() in lowered:
                return lowered[name.lower()]
        return None

    code_col = pick("symbol", "股票代码", "证券代码", "code")
    start_col = pick("start_date", "计入日期", "起始日期", "start")
    industry_col = pick("industry_code", "行业代码", "申万代码")
    name_col = pick("行业名称", "一级行业名称", "industry_name", "name")
    if code_col is None or start_col is None or (industry_col is None and name_col is None):
        raise MarketDataError(f"申万分类历史缺少代码、计入日期或行业。现有列：{list(frame.columns)}")
    out = pd.DataFrame(
        {
            "code": frame[code_col].map(_stock_code),
            "start": pd.to_datetime(frame[start_col], errors="coerce").dt.normalize(),
            "industry_code": "" if industry_col is None else frame[industry_col].map(normalize_sw_code),
            "industry_name": "" if name_col is None else frame[name_col].astype(str).str.strip(),
        }
    ).dropna(subset=["start"])
    out = out.loc[out["code"].str.fullmatch(r"\d{6}")].copy()
    if out.empty:
        raise MarketDataError("申万分类历史里没有可用的股票代码和计入日期。")
    return out.sort_values(["code", "start"]).reset_index(drop=True)


def normalize_sw_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {str(column).strip(): column for column in frame.columns}

    def pick(*names: str):
        for name in names:
            if name in columns:
                return columns[name]
        return None

    code_col = pick("指数代码", "行业代码", "code")
    name_col = pick("指数名称", "行业名称", "name")
    if code_col is None or name_col is None:
        raise MarketDataError(f"申万一级目录缺少代码或名称。现有列：{list(frame.columns)}")
    out = pd.DataFrame(
        {
            "code": frame[code_col].map(normalize_sw_code),
            "name": frame[name_col].astype(str).str.replace("(申万)", "", regex=False).str.strip(),
        }
    )
    out = out.loc[out["code"].str.len() > 0].drop_duplicates("code")
    if out.empty:
        raise MarketDataError("申万一级目录是空的。")
    return out.reset_index(drop=True)


def l1_name(industry_code: str, industry_name: str, catalog: pd.DataFrame) -> str | None:
    names = dict(zip(catalog["code"].astype(str), catalog["name"].astype(str)))
    code = normalize_sw_code(industry_code)
    if code in names:
        return names[code]
    parent = class_l1_index(code)
    if parent and parent in names:
        return names[parent]
    label = str(industry_name or "").replace("(申万)", "").strip()
    catalog_names = set(catalog["name"].astype(str))
    if label in catalog_names:
        return label
    return None


def asof_l1(membership: pd.DataFrame, catalog: pd.DataFrame, code: str, on_date) -> str | None:
    """Industry in effect on on_date. A later reclassification does not apply backwards."""
    stamp = pd.Timestamp(on_date).normalize()
    key = str(code).zfill(6)[-6:]
    rows = membership.loc[(membership["code"] == key) & (membership["start"] <= stamp)]
    if rows.empty:
        return None
    chosen = rows.sort_values("start").iloc[-1]
    return l1_name(str(chosen["industry_code"]), str(chosen["industry_name"]), catalog)


def industry_frame_for_events(events: pd.DataFrame, membership: pd.DataFrame, catalog: pd.DataFrame, indexes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build industry, date, close for the Shenwan level-1 name each event uses."""
    pieces = []
    missing = []
    for event in events.itertuples(index=False):
        name = str(getattr(event, "sw_l1", "") or "")
        if name in {"", "nan", "None"}:
            name = asof_l1(membership, catalog, event.code, event.event_date) or ""
        if not name:
            missing.append(str(event.code).zfill(6)[-6:])
            continue
        panel = indexes.get(name)
        if panel is None or panel.empty:
            missing.append(str(event.code).zfill(6)[-6:])
            continue
        part = panel.copy()
        part["industry"] = name
        pieces.append(part)
    if missing:
        sample = "、".join(missing[:8])
        raise MarketDataError(f"申万一级分类没有覆盖这些代码：{sample}。事件表里的行业名称不会拿来代替。")
    if not pieces:
        raise MarketDataError("没有可用的申万一级行业指数。")
    out = pd.concat(pieces, ignore_index=True)
    return out.loc[:, ["industry", "date", "close"]].drop_duplicates(["industry", "date"]).reset_index(drop=True)


def _read_or_store(path: Path, fetch) -> tuple[pd.DataFrame, bool]:
    if path.exists():
        return pd.read_parquet(path), True
    frame = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise MarketDataError(f"原始文件已存在，拒绝覆盖：{path}")
    frame.to_parquet(path, index=False)
    return frame, False


def _sw_dir(raw_dir: Path) -> Path:
    return Path(raw_dir) / "sw"


def load_sw_catalog(raw_dir: Path, fetch=None) -> tuple[pd.DataFrame, bool]:
    path = _sw_dir(raw_dir) / "l1_catalog.parquet"

    def _fetch():
        if fetch is not None:
            frame = fetch()
        else:
            import akshare as ak

            frame = ak.index_realtime_sw(symbol="一级行业")
        return normalize_sw_catalog(frame)

    return _read_or_store(path, _fetch)


def fetch_sw_membership_frame() -> pd.DataFrame:
    """Download the Shenwan classification history. An SSL failure retries once."""
    import akshare as ak

    try:
        return ak.stock_industry_clf_hist_sw()
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        if "SSL" not in text and "certificate" not in text.lower():
            raise
    import requests

    url = "https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, timeout=180, verify=False, headers=headers)
    response.raise_for_status()
    return pd.read_excel(io.BytesIO(response.content), dtype={"股票代码": str, "行业代码": str})


def load_sw_membership(raw_dir: Path, fetch=None) -> tuple[pd.DataFrame, bool]:
    path = _sw_dir(raw_dir) / "membership.parquet"

    def _fetch():
        frame = fetch() if fetch is not None else fetch_sw_membership_frame()
        return normalize_membership(frame)

    return _read_or_store(path, _fetch)


def load_sw_index(raw_dir: Path, code: str, name: str, fetch=None) -> tuple[pd.DataFrame, bool]:
    path = _sw_dir(raw_dir) / "index" / f"{normalize_sw_code(code)}.parquet"

    def _fetch():
        if fetch is not None:
            frame = fetch(code)
        else:
            import akshare as ak

            frame = ak.index_hist_sw(symbol=normalize_sw_code(code), period="day")
        dated = pd.to_datetime(frame["日期"], errors="coerce").dt.normalize()
        close = pd.to_numeric(frame["收盘"], errors="coerce")
        out = pd.DataFrame({"date": dated, "close": close}).dropna()
        out = out.loc[out["close"] > 0].drop_duplicates("date")
        if out.empty:
            raise MarketDataError(f"申万指数 {name}（{code}）没有收盘价。")
        return out.reset_index(drop=True)

    return _read_or_store(path, _fetch)


def load_sw_industry_frame(events: pd.DataFrame, raw_dir: Path, on_item=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return events labeled with sw_l1, and the level-1 close panel. Existing files are kept."""
    catalog, catalog_hit = load_sw_catalog(raw_dir)
    membership, member_hit = load_sw_membership(raw_dir)
    labeled = events.copy()
    labels = [
        asof_l1(membership, catalog, event.code, event.event_date) for event in labeled.itertuples(index=False)
    ]
    labeled["sw_l1"] = labels
    needed: dict[str, str] = {}
    uncovered = []
    for code, name in zip(labeled["code"].astype(str), labels):
        if not name:
            uncovered.append(str(code).zfill(6)[-6:])
            continue
        match = catalog.loc[catalog["name"] == name]
        if match.empty:
            uncovered.append(str(code).zfill(6)[-6:])
            continue
        needed[name] = str(match.iloc[0]["code"])
    if uncovered:
        sample = "、".join(uncovered[:8])
        raise MarketDataError(f"申万一级分类没有覆盖这些代码：{sample}。事件表里的行业名称不会拿来代替。")
    total = 2 + len(needed)
    done = 0

    def _report(symbol: str, hit: bool) -> None:
        nonlocal done
        done += 1
        if on_item is not None:
            on_item(done, total, symbol, hit)

    _report("申万一级目录", catalog_hit)
    _report("申万分类历史", member_hit)
    indexes: dict[str, pd.DataFrame] = {}
    for name, code in sorted(needed.items()):
        panel, hit = load_sw_index(raw_dir, code, name)
        indexes[name] = panel
        _report(name, hit)
    return labeled, industry_frame_for_events(labeled, membership, catalog, indexes)


def _codes_from_status_frame(frame: pd.DataFrame) -> set[str]:
    if frame is None or len(frame) == 0:
        return set()
    column = None
    for name in frame.columns:
        if str(name).strip() in {"代码", "证券代码", "code"}:
            column = name
            break
    if column is None:
        return set()
    return {str(value).zfill(6)[-6:] for value in frame[column].tolist() if str(value).strip()}


def fetch_trading_status(day: pd.Timestamp) -> dict:
    """Exchange lists for one day. A missing down-limit list is recorded, not invented."""
    import akshare as ak

    stamp = pd.Timestamp(day).strftime("%Y%m%d")
    up = ak.stock_zt_pool_em(date=stamp)
    down_available = True
    try:
        down = ak.stock_zt_pool_dtgc_em(date=stamp)
    except ValueError:
        down = pd.DataFrame()
        down_available = False
    try:
        suspended = ak.stock_tfp_em(date=stamp)
    except Exception:
        suspended = pd.DataFrame()
    return {
        "date": pd.Timestamp(day).strftime("%Y-%m-%d"),
        "up": sorted(_codes_from_status_frame(up)),
        "down": sorted(_codes_from_status_frame(down)),
        "suspended": sorted(_codes_from_status_frame(suspended)),
        "down_available": down_available,
    }


def trading_status(raw_dir: Path, day: pd.Timestamp, fetch=None) -> tuple[dict, bool]:
    path = Path(raw_dir) / "status" / f"{pd.Timestamp(day).strftime('%Y%m%d')}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), True
    try:
        payload = fetch(day) if fetch is not None else fetch_trading_status(day)
    except Exception as exc:
        return {
            "date": pd.Timestamp(day).strftime("%Y-%m-%d"),
            "up": [],
            "down": [],
            "suspended": [],
            "down_available": False,
            "error": str(exc),
        }, False
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), True
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload, False


def annotate_exchange_status(events: pd.DataFrame, raw_dir: Path, on_item=None, fetch=None) -> pd.DataFrame:
    """Add limit_exchange, limit_exchange_known, and suspended. Does not drop rows."""
    out = events.copy()
    out["limit_exchange"] = False
    out["limit_exchange_known"] = False
    out["suspended"] = False
    dates = pd.to_datetime(out["event_date"], errors="coerce").dt.normalize().dropna().unique()
    total = max(len(dates), 1)
    done = 0
    by_day: dict[str, dict] = {}
    for day in dates:
        payload, hit = trading_status(Path(raw_dir), pd.Timestamp(day), fetch=fetch)
        by_day[pd.Timestamp(day).strftime("%Y-%m-%d")] = payload
        done += 1
        if on_item is not None:
            on_item(done, total, f"涨跌停 {pd.Timestamp(day).strftime('%Y-%m-%d')}", hit)
    stamps = pd.to_datetime(out["event_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    codes = out["code"].astype(str).str.zfill(6).str[-6:]
    flags = []
    known = []
    suspended = []
    for stamp, code in zip(stamps, codes):
        payload = by_day.get(stamp)
        if payload is None:
            flags.append(False)
            known.append(False)
            suspended.append(False)
            continue
        hit = code in set(payload.get("up", [])) or code in set(payload.get("down", []))
        flags.append(bool(hit))
        known.append(bool(payload.get("down_available", False)) or bool(payload.get("up")))
        suspended.append(code in set(payload.get("suspended", [])))
    out["limit_exchange"] = flags
    out["limit_exchange_known"] = known
    out["suspended"] = suspended
    return out


def reference_root(cfg_root: Path) -> Path:
    return resolve_raw_dir(cfg_root)


def apply_builtin(cfg, events: pd.DataFrame, industry_frame, factors_frame, on_item=None):
    """Fill Shenwan or CH3 when the caller did not upload a file. Uploads win."""
    notes: dict[str, str] = {}
    raw = reference_root(cfg.root)
    if industry_frame is not None:
        notes["industry_source"] = "upload"
    elif cfg.mode == "akshare" and cfg.model == "industry":
        events, industry_frame = load_sw_industry_frame(events, raw, on_item=on_item)
        notes["industry_source"] = "shenwan_l1"
    if factors_frame is not None:
        notes["factor_source"] = "upload"
    elif cfg.mode == "akshare" and cfg.model == "ff3":
        factors_frame = load_ch3_factors()
        notes["factor_source"] = "ch3"
    if cfg.drop_event_day_limit and cfg.mode == "akshare":
        events = annotate_exchange_status(events, raw, on_item=on_item)
        notes["limit_source"] = "exchange_or_band"
    elif cfg.drop_event_day_limit:
        notes["limit_source"] = "band"
    return events, industry_frame, factors_frame, notes
