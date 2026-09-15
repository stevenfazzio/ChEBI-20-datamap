"""Atomic writes, verified parquet output, retrying HTTP. Shared by every stage."""

import os
import random
import stat
import tempfile
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

import config

RETRY_STATUS = {429, 500, 502, 503, 504}


def _match_mode(tmp_path: str, output_path: Path) -> None:
    # mkstemp creates 0600 and os.replace preserves the source mode, so without this
    # every rewrite would silently tighten permissions on the destination.
    if output_path.exists():
        os.chmod(tmp_path, stat.S_IMODE(output_path.stat().st_mode))
    else:
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp_path, 0o666 & ~umask)


def atomic_write_bytes(output_path: Path, data: bytes) -> None:
    """The destination only ever holds a complete file: write to a sibling tmp, then rename."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=output_path.parent, suffix=output_path.suffix + ".tmp")
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            fh.write(data)
        _match_mode(tmp_path, output_path)
        os.replace(tmp_path, output_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def atomic_write_text(output_path: Path, text: str) -> None:
    atomic_write_bytes(output_path, text.encode("utf-8"))


def write_parquet_safely(df: pd.DataFrame, output_path: Path) -> None:
    """Write df so output_path only ever holds a complete, footer-verified parquet file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".parquet.tmp")
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path, index=False)
        _match_mode(tmp_path, output_path)
        meta = pq.read_metadata(tmp_path)
        assert meta.num_rows == len(df), f"row count {meta.num_rows} != {len(df)}"
        # Arrow schema for top-level names: the parquet schema flattens list columns to "col.list.element".
        missing = set(df.columns) - set(pq.read_schema(tmp_path).names)
        assert not missing, f"columns missing on disk: {missing}"
        os.replace(tmp_path, output_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def validate_stage_output(df: pd.DataFrame, stage_name: str, expected_columns) -> None:
    missing = set(expected_columns) - set(df.columns)
    assert not missing, f"{stage_name}: missing columns {missing}"
    assert len(df) > 0, f"{stage_name}: output is empty"
    print(f"{stage_name}: {len(df)} rows, {len(df.columns)} columns")


def report_delta(stage: str, n_in: int, n_out: int, reason: str) -> None:
    print(f"{stage}: {n_in} in -> {n_out} out ({n_in - n_out} dropped: {reason})")


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    max_retries: int = 4,
    timeout: float = config.REQUEST_TIMEOUT_S,
    **kwargs,
) -> requests.Response:
    """One request with backoff on throttling and server faults; any other 4xx raises immediately.

    Returns the response so callers can read headers (PubChem reports its throttling state in one).
    """
    wait = 0.0
    for attempt in range(max_retries):
        try:
            resp = session.request(method, url, timeout=timeout, **kwargs)
            if resp.status_code in RETRY_STATUS and attempt < max_retries - 1:
                retry_after = resp.headers.get("Retry-After", "")
                backoff = min(2**attempt * 5, 60)
                wait = float(retry_after) if retry_after.isdigit() else backoff
                wait += random.uniform(0, wait * 0.1)
                print(f"{resp.status_code} on {url}: retrying in {wait:.0f}s")
            else:
                resp.raise_for_status()
                return resp
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt == max_retries - 1:
                raise
            wait = min(2**attempt * 5, 60)
            print(f"attempt {attempt + 1}: {e}, retrying in {wait:.0f}s")
        time.sleep(wait)
    raise RuntimeError(f"exhausted retries for {url}")
