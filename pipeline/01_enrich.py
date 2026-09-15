"""Stage 01: PubChem properties per CID + fields derived from the description -> data/corpus.parquet.

PubChem's PUG REST property endpoint is called in batches of CIDs (POST, one JSON per batch cached in
data/pubchem_cache/, so a re-run resumes). The properties give the map its display name (PubChem's title),
formula, weight, XLogP and charge; the description gives the role-derived fields (extract.py). Nothing is
dropped: a CID PubChem does not return keeps null properties and falls back to an IUPAC name or "CID n".
"""

import argparse
import datetime as dt
import json
import re
import time

import pandas as pd
import requests

import config
from extract import derived_fields
from io_utils import atomic_write_text, report_delta, request_with_retry, validate_stage_output, write_parquet_safely

PROPERTY_URL = f"{config.PUBCHEM_REST}/compound/cid/property/{','.join(config.PUBCHEM_PROPERTIES)}/JSON"
THROTTLE_RE = re.compile(r"(\w[\w ]*?) status: (\w+) \((\d+)%\)")


class PubChemClient:
    """Sequential batch fetches under PubChem's usage policy, slowing down when its throttling header asks."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = config.USER_AGENT
        self.last_request = 0.0
        self.extra_wait = 0.0

    def _pace(self) -> None:
        wait = config.PUBCHEM_MIN_INTERVAL_S + self.extra_wait - (time.time() - self.last_request)
        if wait > 0:
            time.sleep(wait)

    def _note_throttle(self, resp: requests.Response) -> None:
        # e.g. "Request Count status: Green (0%), Request Time status: Green (0%), Service status: Yellow (57%)"
        header = resp.headers.get("X-Throttling-Control", "")
        loads = [int(pct) for _, _, pct in THROTTLE_RE.findall(header)]
        worst = max(loads, default=0)
        self.extra_wait = 0.0 if worst < 50 else 1.0 if worst < 75 else 5.0
        if self.extra_wait:
            print(f"  PubChem throttle header {header!r}: pacing +{self.extra_wait:.0f}s")

    def _post(self, cids: list[int]) -> list[dict]:
        self._pace()
        try:
            resp = request_with_retry(
                self.session, "POST", PROPERTY_URL, data={"cid": ",".join(map(str, cids))}, max_retries=5
            )
        finally:
            self.last_request = time.time()
        self._note_throttle(resp)
        return resp.json()["PropertyTable"]["Properties"]

    def fetch(self, cids: list[int]) -> list[dict]:
        """Properties for a batch. A 400/404 means at least one CID is unknown to PubChem, which fails the whole
        batch; the batch is then retried one CID at a time and the unknown ones are simply absent."""
        try:
            return self._post(cids)
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code not in (400, 404):
                raise
            print(f"  batch of {len(cids)} returned {e.response.status_code}; retrying CIDs one at a time")
        out = []
        for cid in cids:
            try:
                out += self._post([cid])
            except requests.HTTPError as e:
                if e.response is None or e.response.status_code not in (400, 404):
                    raise
                print(f"  CID {cid}: {e.response.status_code} (not in PubChem)")
        return out


def fetch_pubchem(cids: list[int], limit: int | None = None) -> pd.DataFrame:
    cache = config.PATHS["pubchem_cache"]
    cache.mkdir(parents=True, exist_ok=True)
    cids = sorted(cids)
    if limit:
        cids = cids[:limit]
    batches = [cids[i : i + config.PUBCHEM_BATCH_SIZE] for i in range(0, len(cids), config.PUBCHEM_BATCH_SIZE)]
    todo = [i for i in range(len(batches)) if not (cache / f"batch_{i:04d}.json").exists()]
    print(
        f"PubChem: {len(cids)} CIDs in {len(batches)} batches; {len(batches) - len(todo)} cached, {len(todo)} to fetch"
    )
    client = PubChemClient()
    t0 = time.time()
    for k, i in enumerate(todo, start=1):
        records = client.fetch(batches[i])
        atomic_write_text(cache / f"batch_{i:04d}.json", json.dumps({"cids": batches[i], "records": records}))
        if k % 10 == 0 or k == len(todo):
            rate = k / (time.time() - t0)
            print(f"  [{k}/{len(todo)}] {rate:.2f} batches/s, ~{(len(todo) - k) / max(rate, 1e-6) / 60:.1f} min left")

    records = []
    for i in range(len(batches)):
        payload = json.loads((cache / f"batch_{i:04d}.json").read_text())
        assert payload["cids"] == batches[i], f"batch_{i:04d}.json was written for a different CID list; delete it"
        records += payload["records"]
    props = pd.DataFrame.from_records(records)
    props = props.rename(columns={"CID": "cid"})
    for col in config.PUBCHEM_PROPERTIES:
        if col not in props.columns:
            props[col] = None
    props["MolecularWeight"] = pd.to_numeric(props["MolecularWeight"], errors="coerce")
    props = props.drop_duplicates("cid").set_index("cid").reindex(cids).reset_index()
    props.columns = ["cid"] + [f"pubchem_{c[0].lower() + c[1:]}" for c in props.columns[1:]]
    return props


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, help="fetch PubChem for only the first N CIDs (smoke test; writes no corpus)")
    args = ap.parse_args()

    molecules = pd.read_parquet(config.PATHS["molecules"])
    props = fetch_pubchem(molecules["cid"].tolist(), limit=args.limit)
    n_missing = int(props["pubchem_title"].isna().sum())
    print(f"PubChem: {len(props) - n_missing} of {len(props)} CIDs returned properties ({n_missing} missing)")
    for col in ("pubchem_xLogP", "pubchem_molecularWeight", "pubchem_charge"):
        print(f"  {col}: {props[col].isna().sum()} null")
    if args.limit:
        print(props.head(10).to_string())
        return
    write_parquet_safely(props, config.PATHS["pubchem"])

    n_in = len(molecules)
    corpus = molecules.merge(props, on="cid", how="left", validate="one_to_one")
    assert len(corpus) == n_in
    derived = pd.DataFrame([derived_fields(d) for d in corpus["description"]])
    corpus = pd.concat([corpus, derived], axis=1)
    corpus["name"] = (
        corpus["pubchem_title"]
        .fillna(corpus["pubchem_iUPACName"])
        .fillna("CID " + corpus["cid"].astype(str))
        .astype(str)
    )
    corpus["url"] = corpus["cid"].map(lambda c: config.PUBCHEM_COMPOUND_URL.format(cid=c))
    corpus["embed_text"] = corpus["description"].str.strip()
    report_delta("enrich", n_in, len(corpus), "left join on cid; nothing dropped")
    validate_stage_output(corpus, "corpus", ["cid", "smiles", "description", "split", "name", "url", "embed_text"])

    n_roles = int((corpus["primary_role"] != "").sum())
    n_org = int((corpus["primary_organism"] != "").sum())
    print(f"roles stated for {n_roles} molecules; metabolite organism for {n_org}")
    print(
        "top organisms:",
        corpus.loc[corpus["primary_organism"] != "", "primary_organism"].value_counts().head(12).to_dict(),
    )
    print(
        "top primary roles:", corpus.loc[corpus["primary_role"] != "", "primary_role"].value_counts().head(12).to_dict()
    )
    print(f"name source: {int(corpus['pubchem_title'].notna().sum())} PubChem titles, rest IUPAC or CID")

    write_parquet_safely(corpus, config.PATHS["corpus"])
    meta = {
        "n_molecules": int(len(corpus)),
        "pubchem_properties": config.PUBCHEM_PROPERTIES,
        "pubchem_missing": n_missing,
        "roles_stated": n_roles,
        "organism_stated": n_org,
        "embed_text": "description, stripped",
        "enriched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_write_text(config.PATHS["corpus_meta"], json.dumps(meta, indent=2))
    print(f"wrote {config.PATHS['corpus']}")


if __name__ == "__main__":
    main()
