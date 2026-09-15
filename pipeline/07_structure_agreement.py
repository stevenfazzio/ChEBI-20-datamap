"""Stage 07: how structural is each map neighbourhood? -> data/structure_agreement.parquet (+ meta).

The map places molecules by their descriptions. This stage measures how far that geometry agrees with chemical
structure, using Morgan fingerprints of the SMILES. For every molecule: the Tanimoto similarity of its nearest
neighbours on the map, in the text-embedding space and in fingerprint space (the ceiling); the share of
neighbours those spaces agree on; and its nearest fingerprint neighbours. Stage 05 colours the map by the
coherence ratio (map Tanimoto over the ceiling) and lists the nearest structural neighbours in the hovercard.
The per-region report reads the labels, so this runs after 04 and before 05. All neighbour searches are exact.
About a minute on a laptop; no GPU, no API.
"""

import argparse
import datetime as dt
import json
import textwrap
import time

import numpy as np
import pandas as pd
import rdkit

import config
from io_utils import atomic_write_text, report_delta, validate_stage_output, write_parquet_safely
from structure import (
    knn_cosine,
    knn_layout,
    morgan_fingerprints,
    neighbour_overlap,
    tanimoto_neighbourhoods,
    unescape_smiles,
)

STAGE = "07 structure_agreement"


def load_inputs(files: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    corpus = pd.read_parquet(config.PATHS["corpus"], columns=["cid", "smiles", "name", "pubchem_heavyAtomCount"])
    cids = corpus["cid"].to_numpy()
    emb = np.load(files["npz"], allow_pickle=True)
    assert (emb["cid"] == cids).all(), "embeddings.npz is not aligned with corpus.parquet"
    lay = np.load(files["umap"], allow_pickle=True)
    assert (lay["cid"] == cids).all(), "umap_coords.npz is not aligned with corpus.parquet"
    labels = pd.read_parquet(files["labels"])
    assert (labels["cid"].to_numpy() == cids).all(), "labels.parquet is not aligned with corpus.parquet"
    return corpus, emb["embeddings"].astype(np.float32), lay["coords"].astype(np.float32), labels


def describe(v: np.ndarray) -> dict:
    q = np.percentile(v, [10, 50, 90])
    return {"mean": round(float(v.mean()), 3), "p10": round(float(q[0]), 3), "median": round(float(q[1]), 3),
            "p90": round(float(q[2]), 3)}  # fmt: skip


def fmt(d: dict) -> str:
    return "  ".join(f"{k} {v:.3f}" for k, v in d.items())


def region_report(out: pd.DataFrame, labels: pd.DataFrame, heavy: np.ndarray) -> dict:
    """Mean coherence per named region and layer. Printed for the mid-granularity layers, stored for all."""
    cols = sorted((c for c in labels.columns if c.startswith("label_layer_")), key=lambda c: int(c.rsplit("_", 1)[1]))
    regions = {}
    for c in cols:
        lab = labels[c].fillna("Unlabelled").to_numpy()
        named = lab != "Unlabelled"
        g = (
            pd.DataFrame(
                {
                    "region": lab[named],
                    "coherence": out["coherence"].to_numpy()[named],
                    "overlap_text_morgan": out["overlap_text_morgan"].to_numpy()[named],
                    "tanimoto_map": out["tanimoto_map"].to_numpy()[named],
                    "heavy_atoms": heavy[named],
                }
            )
            .groupby("region")
            .agg(
                n=("coherence", "size"),
                coherence=("coherence", "mean"),
                overlap_text_morgan=("overlap_text_morgan", "mean"),
                tanimoto_map=("tanimoto_map", "mean"),
                heavy_atoms_median=("heavy_atoms", "median"),
            )
            .sort_values("coherence", ascending=False)
        )
        regions[c] = g.round(3).reset_index().to_dict(orient="records")
        if 10 <= len(g) <= 100:
            print(f"\n{c}: {len(g)} named regions ranked by mean coherence")
            print("  most structural:")
            print(textwrap.indent(g.head(8).round(2).to_string(), "    "))
            print("  least structural:")
            print(textwrap.indent(g.tail(8).round(2).to_string(), "    "))
    return regions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedding", default=config.EMBED_MODEL_KEY, choices=sorted(config.EMBED_MODELS))
    args = ap.parse_args()
    files = config.keyed_files(args.embedding)
    t0 = time.time()

    corpus, embeddings, coords, labels = load_inputs(files)
    n, k = len(corpus), config.AGREEMENT_K
    print(f"{n} molecules, embeddings {embeddings.shape}, coords {coords.shape}, k = {k}")

    fps = morgan_fingerprints(unescape_smiles(corpus["smiles"]))
    n_bits = fps.sum(1)
    n_distinct = len({row.tobytes() for row in fps})
    print(
        f"Morgan r{config.MORGAN_RADIUS}/{config.MORGAN_N_BITS} fingerprints: {n_distinct} distinct, "
        f"{int((n_bits == 0).sum())} empty, bits set median {int(np.median(n_bits))} ({time.time() - t0:.0f}s)"
    )

    idx_text = knn_cosine(embeddings, k)
    idx_map = knn_layout(coords, k)
    idx_random = np.random.default_rng(config.UMAP_RANDOM_STATE).integers(0, n, (n, k))
    print(f"text and layout neighbours found ({time.time() - t0:.0f}s)")
    idx_morgan, sims_morgan, tani = tanimoto_neighbourhoods(
        fps, k, gather={"map": idx_map, "text": idx_text, "random": idx_random}
    )
    tani["morgan"] = sims_morgan.mean(1)
    print(f"fingerprint neighbours found ({time.time() - t0:.0f}s)")

    stored = config.STRUCTURE_NEIGHBOURS_STORED
    cids = corpus["cid"].to_numpy()
    out = pd.DataFrame(
        {
            "cid": cids,
            "tanimoto_map": tani["map"],
            "tanimoto_text": tani["text"],
            "tanimoto_morgan": tani["morgan"],
            "coherence": tani["map"] / np.maximum(tani["morgan"], config.COHERENCE_CEILING_FLOOR),
            "overlap_text_morgan": neighbour_overlap(idx_text, idx_morgan),
            "overlap_map_morgan": neighbour_overlap(idx_map, idx_morgan),
            "overlap_text_map": neighbour_overlap(idx_text, idx_map),
            "morgan_bits": n_bits.astype(np.int16),
            "morgan_neighbour_cids": list(cids[idx_morgan[:, :stored]]),
            "morgan_neighbour_tanimoto": list(sims_morgan[:, :stored].round(3)),
        }
    )

    # ── Report ──
    print("\nneighbourhood overlap: share of the k nearest neighbours two spaces agree on (random ≈ k/n)")
    overlap_summary = {}
    for col in ("overlap_text_morgan", "overlap_map_morgan", "overlap_text_map"):
        v = out[col].to_numpy()
        overlap_summary[col] = describe(v) | {"share_zero": round(float((v == 0).mean()), 3),
                                              "share_half_or_more": round(float((v >= 0.5).mean()), 3)}  # fmt: skip
        print(f"  {col:20s} {fmt(overlap_summary[col])}")
    print("\nmean Tanimoto similarity of a molecule's k neighbours, by the space that chose them")
    tanimoto_summary = {name: describe(tani[name]) for name in ("morgan", "text", "map", "random")}
    for name, d in tanimoto_summary.items():
        print(f"  {name:8s} {fmt(d)}")
    coherence_summary = describe(out["coherence"].to_numpy())
    print(f"\ncoherence (map Tanimoto / fingerprint ceiling, floor {config.COHERENCE_CEILING_FLOOR}):")
    print(f"  {fmt(coherence_summary)}")
    heavy = corpus["pubchem_heavyAtomCount"].to_numpy(dtype=float)
    by_size = out.groupby(pd.qcut(heavy, 4), observed=True)["coherence"].mean()
    print("coherence by heavy-atom quartile: " + ", ".join(f"{iv} {v:.2f}" for iv, v in by_size.items()))
    regions = region_report(out, labels, heavy)

    # ── Write ──
    validate_stage_output(out, STAGE, ["cid", "coherence", "tanimoto_map", "morgan_neighbour_cids"])
    report_delta(STAGE, n, len(out), "nothing dropped")
    write_parquet_safely(out, files["structure"])
    meta = {
        "embedding": args.embedding,
        "fingerprint": {
            "type": "Morgan",
            "radius": config.MORGAN_RADIUS,
            "n_bits": config.MORGAN_N_BITS,
            "chirality": config.MORGAN_CHIRALITY,
            "rdkit": rdkit.__version__,
            "n_distinct": n_distinct,
            "n_empty": int((n_bits == 0).sum()),
        },
        "k": k,
        "coherence_ceiling_floor": config.COHERENCE_CEILING_FLOOR,
        "neighbours_stored": stored,
        "n_molecules": n,
        "summary": {"overlap": overlap_summary, "tanimoto": tanimoto_summary, "coherence": coherence_summary},
        "regions": regions,
        "seconds": round(time.time() - t0),
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_write_text(files["structure_meta"], json.dumps(meta, indent=2))
    print(f"wrote {files['structure']} and {files['structure_meta']} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
