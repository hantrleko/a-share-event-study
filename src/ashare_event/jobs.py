"""Background runs for long AKShare downloads.

The worker lives in this process, not in a Streamlit rerun. Status, the result
pickle, and the zip are files under the job directory. A later page open reads
those files. The directory is ephemeral on a host that redeploys.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path

_LOCK = threading.Lock()
_THREADS: dict[str, threading.Thread] = {}
_JOB_ID = re.compile(r"[0-9a-f]{8,32}")
BACKGROUND_EVENT_THRESHOLD = 30


def should_background(*, n_events: int, mode: str, source: str, forced: bool) -> bool:
    """Demo and small runs stay on the page. A checked box, or a long download, leaves the page."""
    if forced:
        return True
    if source == "演示样本":
        return False
    return str(mode) == "akshare" and int(n_events) >= BACKGROUND_EVENT_THRESHOLD


def jobs_root(base: Path | None = None) -> Path:
    override = os.environ.get("ASHARE_JOB_DIR", "").strip()
    root = Path(override) if override else Path(base or Path.cwd()) / "data" / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def prepare_job(base: Path | None = None, meta: dict | None = None) -> str:
    job_id = uuid.uuid4().hex[:12]
    folder = _folder(job_id, base)
    folder.mkdir(parents=True, exist_ok=False)
    status = {
        "state": "running",
        "done": 0,
        "total": 0,
        "label": "排队",
        "message": "",
        "started": _now(),
    }
    if meta:
        status.update(_jsonable(meta))
    _write_status(folder, status)
    return job_id


def execute_job(job_id: str, worker, *, base: Path | None = None) -> None:
    """Run ``worker(progress)`` in the current thread and write status files."""
    folder = _folder(job_id, base)

    def progress(done, total, label) -> None:
        changes = {}
        if done is not None:
            changes["done"] = int(done)
        if total is not None:
            changes["total"] = int(total)
        if label:
            changes["label"] = str(label)
        _update(folder, **changes)

    try:
        result = worker(progress)
    except Exception as exc:
        _update(folder, state="error", message=str(exc), finished=_now(), label="没有跑完")
        return
    _store_result(folder, result)
    _update(folder, state="complete", label="计算完成", finished=_now(), **_summary(result))


def submit_job(worker, *, base: Path | None = None, meta: dict | None = None, context=None) -> str:
    """Start a daemon thread. The Streamlit rerun does not own the thread."""
    job_id = prepare_job(base, meta)
    if context is not None:
        with (_folder(job_id, base) / "context.pkl").open("wb") as handle:
            pickle.dump(context, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _target() -> None:
        try:
            execute_job(job_id, worker, base=base)
        finally:
            with _LOCK:
                _THREADS.pop(job_id, None)

    thread = threading.Thread(target=_target, name=f"ashare-job-{job_id}", daemon=True)
    with _LOCK:
        _THREADS[job_id] = thread
    thread.start()
    return job_id


def job_status(job_id: str, base: Path | None = None) -> dict:
    path = _folder(job_id, base) / "status.json"
    if not path.exists():
        return {"state": "missing", "message": "没有这个任务。"}
    with _LOCK:
        data = _read_status(path.parent)
        if data.get("state") == "running" and not _alive(job_id):
            data = _read_status(path.parent)
        if data.get("state") == "running" and not _alive(job_id):
            data["state"] = "lost"
            data["message"] = "后台线程不在了。请重新运行。"
            _write_status(path.parent, data)
        return dict(data)


def load_job_result(job_id: str, base: Path | None = None):
    path = _folder(job_id, base) / "result.pkl"
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_job_context(job_id: str, base: Path | None = None):
    path = _folder(job_id, base) / "context.pkl"
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def job_zip_bytes(job_id: str, base: Path | None = None) -> bytes | None:
    path = _folder(job_id, base) / "result.zip"
    if not path.exists():
        return None
    return path.read_bytes()


def rewrite_job_zip(job_id: str, result, base: Path | None = None) -> None:
    folder = _folder(job_id, base)
    if not folder.exists():
        return
    from ashare_event.bundle import build_zip

    (folder / "result.zip").write_bytes(build_zip(result))


def _folder(job_id: str, base: Path | None) -> Path:
    if not _JOB_ID.fullmatch(str(job_id or "")):
        raise ValueError("任务编号不对。")
    return jobs_root(base) / str(job_id)


def _alive(job_id: str) -> bool:
    thread = _THREADS.get(job_id)
    return thread is not None and thread.is_alive()


def _store_result(folder: Path, result) -> None:
    try:
        with (folder / "result.pkl").open("wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        (folder / "pickle_error.txt").write_text(str(exc), encoding="utf-8")
    if hasattr(result, "config") and hasattr(result, "tests"):
        try:
            from ashare_event.bundle import build_zip

            (folder / "result.zip").write_bytes(build_zip(result))
        except Exception as exc:
            (folder / "zip_error.txt").write_text(str(exc), encoding="utf-8")


def _summary(result) -> dict:
    if not hasattr(result, "tests") or not hasattr(result, "config"):
        return {}
    try:
        cfg = result.config
        window = (int(cfg.regression_window[0]), int(cfg.regression_window[1]))
        matched = result.tests[(result.tests["t1"] == window[0]) & (result.tests["t2"] == window[1])]
        payload = {
            "model": cfg.model,
            "car_window": [window[0], window[1]],
            "simulated": bool(getattr(result, "simulated", False)),
        }
        if matched.empty:
            return payload
        row = matched.iloc[0]
        preferred = str(row["preferred"])
        t_value = None
        if preferred in {"portfolio", "kp", "bmp"}:
            t_value = _finite(row[f"t_{preferred}"])
        payload.update(
            {
                "n": int(row["n"]),
                "mean_car_pp": _finite(float(row["mean"]) * float(cfg.return_scale)),
                "preferred": preferred,
                "t_preferred": t_value,
            }
        )
        return payload
    except Exception:
        return {}


def _update(folder: Path, **changes) -> None:
    with _LOCK:
        current = _read_status(folder)
        if current.get("state") in {"complete", "error", "lost"} and "state" not in changes:
            return
        current.update(_jsonable(changes))
        _write_status(folder, current)


def _read_status(folder: Path) -> dict:
    path = folder / "status.json"
    if not path.exists():
        return {"state": "missing", "message": "没有这个任务。"}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_status(folder: Path, payload: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "status.json"
    temporary = folder / "status.json.tmp"
    temporary.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _jsonable(payload: dict) -> dict:
    clean = {}
    for key, value in payload.items():
        if isinstance(value, float):
            clean[key] = value if _finite(value) is not None else None
        else:
            clean[key] = value
    return clean


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
