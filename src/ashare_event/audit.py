"""Row-level checks for an event grid before estimation.

Errors should stop a run. Warnings and notes do not drop rows.
"""

from __future__ import annotations

import pandas as pd

from ashare_event.schema import SchemaError, normalize_code

_PREFIXES = (
    "000",
    "001",
    "002",
    "003",
    "300",
    "301",
    "302",
    "600",
    "601",
    "603",
    "605",
    "688",
    "689",
    "430",
    "920",
)


def known_listing(code: str) -> bool:
    text = str(code).zfill(6)[-6:]
    if text.startswith(_PREFIXES):
        return True
    return text.startswith(("8", "4"))


def audit_grid(frame: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["行号", "代码", "级别", "问题"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    rows: list[tuple] = []
    seen: dict[tuple, int] = {}
    for pos, row in enumerate(frame.itertuples(index=False), start=1):
        raw_code = getattr(row, "code", "")
        code_text = "" if raw_code is None else str(raw_code).strip()
        if code_text.lower() == "nan":
            code_text = ""
        raw_date = getattr(row, "event_date", None)
        parsed = pd.to_datetime(raw_date, errors="coerce")
        date_missing = raw_date is None or pd.isna(parsed) or str(raw_date).strip() in {"", "None", "nan", "NaT"}
        if not code_text and date_missing:
            continue
        if not code_text:
            rows.append((pos, "", "错误", "有事件日期，但证券代码是空的。"))
            continue
        try:
            norm = normalize_code(code_text)
        except SchemaError:
            rows.append((pos, code_text, "错误", "证券代码无法解析。请写成 6 位数字，或 SH600519 这种形式。"))
            continue
        if not known_listing(norm):
            rows.append((pos, norm, "警告", "代码前缀不在常见的沪深京范围内。请核对是不是输错了。"))
        if date_missing:
            rows.append((pos, norm, "错误", "事件日期是空的或无法识别。请写成 2024-03-18。"))
            continue
        if parsed.dayofweek >= 5:
            rows.append((pos, norm, "提示", "这一天是周末。运行时会顺延到其后 10 个日历日内的第一个交易日。"))
        firm = str(getattr(row, "firm_id", "") or "").strip()
        identity = norm if firm.lower() in {"", "nan", "none"} else firm
        key = (identity, pd.Timestamp(parsed).normalize())
        if key in seen:
            rows.append((pos, norm, "错误", f"与第 {seen[key]} 行是同一公司、同一天。请合并或删掉一行。"))
        else:
            seen[key] = pos
        name = str(getattr(row, "firm_name", "") or "")
        if "示例" in name:
            rows.append((pos, norm, "错误", "公司名里还有「示例」。这是教学行，核对后删掉这两个字再运行。"))
    return pd.DataFrame(rows, columns=columns)
