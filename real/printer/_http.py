"""Tiny stdlib HTTP helpers (no requests dependency)."""
from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from real.printer.base import PrinterError


def get(url: str, headers: dict | None = None, timeout: float = 5.0) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise PrinterError(f"HTTP {e.code} for {url}: {e.read()[:200]!r}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise PrinterError(f"cannot reach {url}: {getattr(e, 'reason', e)}") from e


def get_json(url: str, headers: dict | None = None, timeout: float = 5.0):
    body = get(url, headers, timeout)
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise PrinterError(f"not JSON from {url}: {body[:120]!r}") from e


def download(url: str, dest: str | Path, headers: dict | None = None, timeout: float = 60.0) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
    except urllib.error.HTTPError as e:
        raise PrinterError(f"HTTP {e.code} downloading {url}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise PrinterError(f"cannot download {url}: {getattr(e, 'reason', e)}") from e
    return dest
