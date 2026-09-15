"""Stage 00: fetch the three ChEBI-20 splits -> data/raw/*.txt, data/molecules.parquet (+ raw/_manifest.json).

The raw files are the provenance record: a split file that already exists is not refetched (delete it to
refetch). Rows are parsed exactly as published (tab-separated, no quoting); SMILES are kept verbatim.
"""

import csv
import datetime as dt
import hashlib
import json

import pandas as pd
import requests

import config
from io_utils import atomic_write_bytes, atomic_write_text, report_delta, request_with_retry, write_parquet_safely


def fetch_split(session: requests.Session, split: str) -> dict:
    path = config.RAW_DIR / f"{split}.txt"
    url = f"{config.CHEBI20_BASE_URL}/{split}.txt"
    if path.exists():
        print(f"{path.name}: exists, not refetched")
    else:
        resp = request_with_retry(session, "GET", url)
        atomic_write_bytes(path, resp.content)
        print(f"{path.name}: fetched {len(resp.content) / 1e6:.1f} MB")
    return {"url": url, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def parse_split(split: str) -> pd.DataFrame:
    rows, malformed = [], 0
    with open(config.RAW_DIR / f"{split}.txt", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        assert header == ["CID", "SMILES", "description"], f"{split}: unexpected header {header}"
        for row in reader:
            if len(row) != 3:
                malformed += 1
                continue
            rows.append(row)
    df = pd.DataFrame(rows, columns=["cid", "smiles", "description"])
    df["cid"] = df["cid"].astype("int64")
    df["split"] = split
    report_delta(f"parse {split}", len(rows) + malformed, len(df), "malformed rows (not three tab-separated fields)")
    expected = config.CHEBI20_EXPECTED_ROWS[split]
    assert len(df) == expected, f"{split}: {len(df)} rows, expected {expected}"
    return df


def main() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = config.USER_AGENT
    files = {split: fetch_split(session, split) for split in config.CHEBI20_SPLITS}

    frames = [parse_split(split) for split in config.CHEBI20_SPLITS]
    molecules = pd.concat(frames, ignore_index=True)
    n_before = len(molecules)
    assert molecules["cid"].is_unique, "CIDs repeat across splits"
    molecules["n_words"] = molecules["description"].str.split().str.len().astype("int32")
    assert (molecules["n_words"] >= 1).all() and (molecules["description"].str.len() > 0).all()
    report_delta("merge splits", n_before, len(molecules), "nothing dropped; every CID is unique")
    print(
        f"{len(molecules)} molecules; description words: median {molecules['n_words'].median():.0f}, "
        f"p90 {molecules['n_words'].quantile(0.9):.0f}, max {molecules['n_words'].max()}"
    )

    write_parquet_safely(molecules, config.PATHS["molecules"])
    manifest = {
        "source": "ChEBI-20 (Edwards et al. 2022, MolT5), files as published in blender-nlp/MolT5",
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "files": files,
        "rows": {split: int(len(f)) for split, f in zip(config.CHEBI20_SPLITS, frames)},
        "columns": ["CID", "SMILES", "description"],
    }
    atomic_write_text(config.PATHS["manifest"], json.dumps(manifest, indent=2))
    print(f"wrote {config.PATHS['molecules']} and {config.PATHS['manifest']}")


if __name__ == "__main__":
    main()
