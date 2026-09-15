"""Stage 08: reaction context from Rhea -> data/rhea.parquet (+ rhea_meta.json); raw files in data/raw/rhea/.

Rhea is the expert-curated reaction database whose participants are ChEBI entities, so it is the natural source of
measured biochemistry for a ChEBI-derived corpus. Molecules are matched to Rhea participants by the connectivity
layer of their InChIKey (ChEBI-20 lists many compounds in both neutral and charged forms; Rhea uses the major species
at pH 7.3). Per molecule: the master reactions it takes part in, whether it is a cofactor hub, the dominant enzyme
class among its reactions, and its reaction partners inside ChEBI-20 (the other participants in the same reactions,
hubs and other charge states of the same compound excluded), ranked by shared reactions. Stage 05 colours both maps
by enzyme class and lists partners in the hovercard. The report scores partners in the text and fingerprint spaces,
which says which map predicts biochemical adjacency.

Layout-independent: runs after 01 (corpus) and 02 (embeddings, for the report) and before 05. The raw files are the
provenance record and are not refetched while present; delete them to refetch a newer Rhea release.
"""

import collections
import datetime as dt
import gzip
import hashlib
import json
import re
import time

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, RDLogger

import config
from io_utils import (
    atomic_write_bytes,
    atomic_write_text,
    report_delta,
    request_with_retry,
    validate_stage_output,
    write_parquet_safely,
)
from structure import cosine_neighbourhoods, morgan_fingerprints, tanimoto_neighbourhoods, unescape_smiles

STAGE = "08 rhea"


def fetch_rhea(session: requests.Session) -> dict:
    config.RHEA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rel in config.RHEA_FILES.items():
        path, url = config.RHEA_RAW_DIR / name, f"{config.RHEA_BASE_URL}/{rel}"
        if path.exists():
            print(f"{name}: exists, not refetched")
        else:
            resp = request_with_retry(session, "GET", url)
            atomic_write_bytes(path, resp.content)
            print(f"{name}: fetched {len(resp.content) / 1e6:.1f} MB")
        files[name] = {
            "url": url,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return files


def parse_release() -> dict:
    props = dict(
        line.strip().split("=", 1)
        for line in (config.RHEA_RAW_DIR / "rhea-release.properties").read_text().splitlines()
        if "=" in line and not line.startswith("#")
    )
    return {k: v for k, v in props.items() if "release" in k.lower()}


def parse_master_reactions(path) -> list[tuple[str, list[str]]]:
    """Rhea lists every reaction four times (undirected, both directions, bidirectional); keep the undirected master
    ("=" with no arrow) and its participants as ChEBI ids."""
    reactions, entry, equation = [], None, None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("ENTRY"):
                entry = line.split()[1]
            elif line.startswith("EQUATION"):
                equation = line[12:].strip()
            elif line.startswith("///"):
                if equation and " = " in equation and "=>" not in equation and "<=" not in equation:
                    reactions.append((entry, sorted(set(re.findall(r"CHEBI:\d+", equation)))))
                entry, equation = None, None
    return reactions


def connectivity_blocks(smiles) -> list[str | None]:
    """The first 14 characters of the InChIKey: the molecular skeleton without stereo, charge or isotopes."""
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        out.append(Chem.MolToInchiKey(mol)[:14] if mol is not None else None)
    return out


def rank_partners(shared: dict[int, int], degree: dict[int, int]) -> list[int]:
    """Partners by reactions shared (desc), then how promiscuous the partner is (asc), then index."""
    return sorted(shared, key=lambda j: (-shared[j], degree.get(j, 0), j))


def dominant_ec_class(classes: collections.Counter) -> str:
    """The commonest top-level EC class among a molecule's reactions; ties go to the lower class number."""
    if not classes:
        return "Unassigned"
    top = max(classes.values())
    return config.EC_CLASSES[min(c for c, n in classes.items() if n == top)]


def main() -> None:
    t0 = time.time()
    session = requests.Session()
    session.headers["User-Agent"] = config.USER_AGENT
    files = fetch_rhea(session)
    release = parse_release()
    print(f"Rhea release: {release}")

    corpus = pd.read_parquet(config.PATHS["corpus"], columns=["cid", "smiles", "name"])
    n = len(corpus)
    cids = corpus["cid"].to_numpy()
    blocks = connectivity_blocks(unescape_smiles(corpus["smiles"]))
    print(f"{n} molecules, {sum(b is not None for b in blocks)} with an InChIKey ({time.time() - t0:.0f}s)")

    rhea = pd.read_csv(config.RHEA_RAW_DIR / "rhea-chebi-smiles.tsv", sep="\t", header=None, names=["chebi", "smiles"])
    rhea["block"] = connectivity_blocks(rhea["smiles"])
    by_block = collections.defaultdict(list)
    for chebi, block in zip(rhea["chebi"], rhea["block"]):
        if block:
            by_block[block].append(chebi)
    chebi_to_idx = collections.defaultdict(list)
    matched = [by_block.get(b, []) for b in blocks]
    for i, chebis in enumerate(matched):
        for c in chebis:
            chebi_to_idx[c].append(i)
    reactions = parse_master_reactions(config.RHEA_RAW_DIR / "rhea-reactions.txt.gz")
    degree = collections.Counter(c for _, parts in reactions for c in parts)
    hubs = {c for c, d in degree.items() if d > config.RHEA_HUB_MIN_REACTIONS}
    print(
        f"Rhea: {len(rhea)} participants, {len(reactions)} master reactions, {len(hubs)} hubs (> "
        f"{config.RHEA_HUB_MIN_REACTIONS} reactions); {sum(bool(m) for m in matched)} molecules matched "
        f"({sum(bool(m) for m in matched) / n:.1%}) ({time.time() - t0:.0f}s)"
    )

    ec = pd.read_csv(config.RHEA_RAW_DIR / "rhea2ec.tsv", sep="\t", dtype=str)
    reaction_classes = collections.defaultdict(set)
    for master, number in zip(ec["MASTER_ID"], ec["ID"]):
        reaction_classes[f"RHEA:{master}"].add(number.split(".")[0])

    # Per molecule: reactions, enzyme classes, and partners (other non-hub ChEBI-20 participants of the same reactions,
    # not the same skeleton in another charge state). Everything runs over master reactions, so a pair sharing a
    # reaction is counted once.
    n_reactions = np.zeros(n, dtype=np.int32)
    is_hub = np.zeros(n, dtype=bool)
    classes = [collections.Counter() for _ in range(n)]
    shared = [collections.Counter() for _ in range(n)]
    for rid, parts in reactions:
        members = {i: c for c in parts for i in chebi_to_idx.get(c, [])}
        for i, c in members.items():
            n_reactions[i] += 1
            if c in hubs:
                is_hub[i] = True
        nonhub = [i for i, c in members.items() if c not in hubs]
        for i in nonhub:
            classes[i].update(reaction_classes.get(rid, ()))
            for j in nonhub:
                if j != i and blocks[j] != blocks[i]:
                    shared[i][j] += 1
    in_rhea = n_reactions > 0
    ec_class = np.array(
        ["Cofactor hub" if is_hub[i] else dominant_ec_class(classes[i]) if in_rhea[i] else "" for i in range(n)],
        dtype=object,
    )
    partner_lists = [
        rank_partners(shared[i], {j: int(n_reactions[j]) for j in shared[i]}) if not is_hub[i] else [] for i in range(n)
    ]
    out = pd.DataFrame(
        {
            "cid": cids,
            "in_rhea": in_rhea,
            "rhea_chebi_ids": [list(m) for m in matched],
            "n_reactions": n_reactions,
            "is_hub": is_hub,
            "ec_class": ec_class,
            "n_partners": np.array([len(p) for p in partner_lists], dtype=np.int32),
            "partner_cids": [list(cids[p]) for p in partner_lists],
            "partner_shared": [[int(shared[i][j]) for j in p] for i, p in enumerate(partner_lists)],
        }
    )

    # ── Report ──
    with_partners = out["n_partners"] > 0
    print(
        f"\nin a Rhea reaction: {in_rhea.sum()} ({in_rhea.mean():.1%}); cofactor hubs among them: {is_hub.sum()}; "
        f"with >= 1 partner inside ChEBI-20: {with_partners.sum()} ({with_partners.mean():.1%})"
    )
    print("hubs in ChEBI-20:", ", ".join(corpus["name"][is_hub].head(20)))
    deg = out.loc[with_partners, "n_partners"]
    print(f"partners per molecule: median {deg.median():.0f}, p90 {deg.quantile(0.9):.0f}, max {deg.max()}")
    print("enzyme class of molecules in Rhea:")
    for cls, count in out.loc[in_rhea, "ec_class"].value_counts().items():
        print(f"  {count:6d}  {cls}")

    # Which map predicts biochemical adjacency: partners scored in the text and fingerprint spaces.
    k = config.AGREEMENT_K
    emb = np.load(config.keyed_files()["npz"], allow_pickle=True)
    assert (emb["cid"] == cids).all(), "embeddings.npz is not aligned with corpus.parquet"
    embeddings = emb["embeddings"].astype(np.float32)
    fps = morgan_fingerprints(unescape_smiles(corpus["smiles"]))
    idx_text, _, _ = cosine_neighbourhoods(embeddings, k)
    idx_morgan, _, _ = tanimoto_neighbourhoods(fps, k)
    F = fps.astype(np.float32)
    cnt = F.sum(1)
    rng = np.random.default_rng(config.UMAP_RANDOM_STATE)
    rows = []
    for i in np.flatnonzero(with_partners.to_numpy()):
        js = np.array(partner_lists[i])
        rnd = rng.integers(0, n, len(js))
        inter = F[js] @ F[i]
        rows.append(
            {
                "in_morgan_knn": float(np.isin(js, idx_morgan[i]).mean()),
                "in_text_knn": float(np.isin(js, idx_text[i]).mean()),
                "tanimoto": float((inter / np.maximum(cnt[js] + cnt[i] - inter, 1)).mean()),
                "tanimoto_random": float(((F[rnd] @ F[i]) / np.maximum(cnt[rnd] + cnt[i] - (F[rnd] @ F[i]), 1)).mean()),
                "cosine": float((embeddings[js] @ embeddings[i]).mean()),
                "cosine_random": float((embeddings[rnd] @ embeddings[i]).mean()),
            }
        )
    referee = pd.DataFrame(rows).mean().round(3).to_dict()
    print(
        f"\npartners as neighbours (means over {len(rows)} molecules): among the {k} fingerprint neighbours "
        f"{referee['in_morgan_knn']:.1%}, among the {k} description neighbours {referee['in_text_knn']:.1%}"
    )
    print(
        f"  Tanimoto to partners {referee['tanimoto']:.3f} (random {referee['tanimoto_random']:.3f}); "
        f"description cosine to partners {referee['cosine']:.3f} (random {referee['cosine_random']:.3f})"
    )

    # ── Write ──
    validate_stage_output(out, STAGE, ["cid", "in_rhea", "n_reactions", "ec_class", "partner_cids"])
    report_delta(STAGE, n, len(out), "nothing dropped; molecules outside Rhea keep empty context")
    write_parquet_safely(out, config.PATHS["rhea"])
    manifest = {
        "source": "Rhea (expert-curated reaction database, https://www.rhea-db.org), public files",
        "release": release,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "files": files,
    }
    atomic_write_text(config.PATHS["rhea_manifest"], json.dumps(manifest, indent=2))
    meta = {
        "release": release,
        "matching": "InChIKey connectivity layer (first 14 characters)",
        "hub_min_reactions": config.RHEA_HUB_MIN_REACTIONS,
        "n_participants": int(len(rhea)),
        "n_master_reactions": len(reactions),
        "n_hubs": len(hubs),
        "n_molecules": n,
        "n_in_rhea": int(in_rhea.sum()),
        "n_hubs_in_corpus": int(is_hub.sum()),
        "n_with_partners": int(with_partners.sum()),
        "ec_class_counts": out.loc[in_rhea, "ec_class"].value_counts().to_dict(),
        "partners_scored": {"k": k, **referee},
        "seconds": round(time.time() - t0),
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_write_text(config.PATHS["rhea_meta"], json.dumps(meta, indent=2))
    print(f"wrote {config.PATHS['rhea']} and {config.PATHS['rhea_meta']} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
