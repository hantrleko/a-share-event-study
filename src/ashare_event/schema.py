"""Event-list schema. Codes are normalised; duplicate firm-dates are rejected."""

from __future__ import annotations

import re

import pandas as pd

REQUIRED_EVENT_COLUMNS = ("code", "event_date")
OPTIONAL_DEFAULTS = {
    "event_type": "unspecified",
    "industry": "unspecified",
    "firm_name": "",
}


class SchemaError(ValueError):
    """The event list cannot be used as a firm-event sample."""


def normalize_code(value) -> str:
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    digits = re.sub(r"\D", "", text)
    if not digits:
        raise SchemaError(f"证券代码无法解析：{value!r}")
    return digits.zfill(6)[-6:]


def load_events(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_EVENT_COLUMNS if col not in frame.columns]
    if missing:
        raise SchemaError(f"事件表缺少列：{missing}。至少需要 code 与 event_date。")
    df = frame.copy()
    df["code"] = df["code"].map(normalize_code)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.normalize()
    if df["event_date"].isna().any():
        raise SchemaError("event_date 存在无法解析的日期。")
    if "firm_id" not in df.columns:
        df["firm_id"] = df["code"]
    df["firm_id"] = df["firm_id"].astype(str)
    for column, default in OPTIONAL_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default
        df[column] = df[column].fillna(default).astype(str)
    df["event_type"] = df["event_type"].map(_canon_event_type).str.strip().replace({"": "unspecified"})
    df["industry"] = df["industry"].str.strip().replace({"": "unspecified"})
    if df.duplicated(["firm_id", "event_date"]).any():
        dup = df.loc[df.duplicated(["firm_id", "event_date"], keep=False), ["firm_id", "event_date"]]
        raise SchemaError(f"同一公司同一天有重复事件，请先合并：\n{dup.head(10)}")
    df = df.sort_values(["event_date", "firm_id"]).reset_index(drop=True)
    df["event_id"] = [
        f"{firm}_{date:%Y%m%d}"
        for firm, date in zip(df["firm_id"], df["event_date"], strict=True)
    ]
    if df["event_id"].duplicated().any():
        raise SchemaError("event_id 冲突。firm_id 与 event_date 的组合必须唯一。")
    return df


def read_events_csv(path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": str, "firm_id": str})
    return load_events(frame)


STANDARD_FIELDS = ("code", "event_date", "event_type", "industry", "firm_name")
FIELD_LABELS = {
    "code": "证券代码",
    "event_date": "事件日期",
    "event_type": "事件类型",
    "industry": "行业",
    "firm_name": "公司名称",
}
FIELD_ALIASES = {
    "code": ["code", "股票代码", "证券代码", "代码", "symbol"],
    "event_date": ["event_date", "事件日期", "公告日期", "日期", "date"],
    "event_type": ["event_type", "事件类型", "类型", "type"],
    "industry": ["industry", "行业", "申万行业", "所属行业"],
    "firm_id": ["firm_id", "公司代码", "firm"],
    "firm_name": ["firm_name", "公司名称", "证券简称", "简称", "名称"],
}
_COVARIATE_ALIASES = {
    "信号成本": "signal_cost",
    "合法性缺口": "legitimacy_gap",
    "对数规模": "size_log",
    "规模": "size_log",
}


def guess_roles(columns: list[str]) -> dict[str, str | None]:
    """Match a file's headers onto the standard event fields."""
    lookup = {str(column).strip().lower(): column for column in columns}
    roles: dict[str, str | None] = {}
    for role, aliases in FIELD_ALIASES.items():
        found = None
        for alias in aliases:
            if alias in columns:
                found = alias
                break
            found = lookup.get(alias.lower())
            if found is not None:
                break
        roles[role] = found
    return roles


def rename_to_standard(frame: pd.DataFrame, roles: dict[str, str | None] | None = None) -> pd.DataFrame:
    """Rename headers onto the standard fields and keep every other column.

    Other columns are user variables. They are not dropped.
    """
    if roles is None:
        roles = guess_roles([str(column) for column in frame.columns])
    if not roles.get("code") or not roles.get("event_date"):
        raise SchemaError("标准事件表需要证券代码和事件日期。请使用这两列的标准名称，或常见中文列名。")
    rename: dict[str, str] = {}
    used: set[str] = set()
    for role, source in roles.items():
        if source is None:
            continue
        source = str(source).strip()
        if not source or source.startswith("（"):
            continue
        if source not in frame.columns:
            raise SchemaError(f"找不到列：{source}")
        if source in used:
            raise SchemaError(f"列「{source}」不能同时对应两个变量。")
        used.add(source)
        if source != role:
            rename[source] = role
    renamed = frame.rename(columns=rename)
    alias_rename = {}
    for source, target in _COVARIATE_ALIASES.items():
        if source in renamed.columns and target not in renamed.columns:
            alias_rename[source] = target
    if alias_rename:
        renamed = renamed.rename(columns=alias_rename)
    if "event_type" in renamed.columns:
        renamed["event_type"] = renamed["event_type"].map(_canon_event_type)
    return renamed


def apply_column_roles(frame: pd.DataFrame, roles: dict[str, str | None]) -> pd.DataFrame:
    """Rename the user's own column names onto the event-study fields.

    ``roles`` maps a standard field (``code``, ``event_date``, ``event_type``,
    ``industry``, ``firm_id``, ``firm_name``) to a column in ``frame``.
    Empty values leave that field unset. Other columns are kept, so a
    user-named numeric variable can still enter the regression.
    """
    return load_events(rename_to_standard(frame, roles))


def _canon_event_type(value) -> str:
    """Map a few common labels onto the lenses. Unknown labels stay as written."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    key = text.lower().replace(" ", "")
    table = {
        "ai披露": "ai_disclosure",
        "人工智能披露": "ai_disclosure",
        "人工智能": "ai_disclosure",
        "aidisclosure": "ai_disclosure",
        "ai_disclosure": "ai_disclosure",
        "esg": "esg",
        "地缘政治": "geopolitics",
        "地缘": "geopolitics",
        "geopolitics": "geopolitics",
    }
    return table.get(key, text)
