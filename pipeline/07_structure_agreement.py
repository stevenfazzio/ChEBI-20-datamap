"""Stage 07: how far does a map's geometry agree with the space it does not show? -> data/structure_agreement.parquet
(+ meta), or the `_morgan` versions for the structure map.

Each map places molecules by one thing: the text map by their descriptions, the structure map by Morgan fingerprints
of their SMILES. This stage scores every molecule's nearest map neighbours by the *other* similarity (Tanimoto of
fingerprints for the text map, cosine of description embeddings for the structure map), relative to chance and to
the best that other space offers, and finds its nearest neighbours in that other space. Stage 05 colours the map by
the resulting coherence and lists those neighbours in the hovercard. The per-region reports read the map's labels,
so this runs after 04 and before 05; the cross-map report needs the other map's labels too and is skipped without
them. All neighbour searches are exact. About a minute on a laptop; no GPU, no API.
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
    cosine_neighbourhoods,
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
    assert (lay["cid"] == cids).all(), f"{files['umap'].name} is not aligned with corpus.parquet"
    labels = pd.read_parquet(files["labels"])
    assert (labels["cid"].to_numpy() == cids).all(), f"{files['labels'].name} is not aligned with corpus.parquet"
    return corpus, emb["embeddings"].astype(np.float32), lay["coords"].astype(np.float32), labels


def describe(v: np.ndarray) -> dict:
    q = np.percentile(v, [10, 50, 90])
    return {"mean": round(float(v.mean()), 3), "p10": round(float(q[0]), 3), "median": round(float(q[1]), 3),
            "p90": round(float(q[2]), 3)}  # fmt: skip


def fmt(d: dict) -> str:
    return "  ".join(f"{k} {v:.3f}" for k, v in d.items())


def layer_columns(df: pd.DataFrame) -> list[str]:
    return sorted((c for c in df.columns if c.startswith("label_layer_")), key=lambda c: int(c.rsplit("_", 1)[1]))


def layer_closest_to(df: pd.DataFrame, target: int) -> str:
    """The label_layer_i column whose number of named groups is closest to `target`."""
    return min(layer_columns(df), key=lambda c: abs(df[c].nunique() - target))


def region_report(out: pd.DataFrame, labels: pd.DataFrame, heavy: np.ndarray) -> dict:
    """Mean coherence per named region and layer. Printed for the mid-granularity layers, stored for all."""
    regions = {}
    for c in layer_columns(labels):
        lab = labels[c].fillna("Unlabelled").to_numpy()
        named = lab != "Unlabelled"
        g = (
            pd.DataFrame(
                {
                    "region": lab[named],
                    "coherence": out["coherence"].to_numpy()[named],
                    "overlap_text_morgan": out["overlap_text_morgan"].to_numpy()[named],
                    "sim_map": out["sim_map"].to_numpy()[named],
                    "heavy_atoms": heavy[named],
                }
            )
            .groupby("region")
            .agg(
                n=("coherence", "size"),
                coherence=("coherence", "mean"),
                overlap_text_morgan=("overlap_text_morgan", "mean"),
                sim_map=("sim_map", "mean"),
                heavy_atoms_median=("heavy_atoms", "median"),
            )
            .sort_values("coherence", ascending=False)
        )
        regions[c] = g.round(3).reset_index().to_dict(orient="records")
        if 10 <= len(g) <= 100:
            print(f"\n{c}: {len(g)} named regions ranked by mean coherence")
            print("  most coherent:")
            print(textwrap.indent(g.head(8).round(2).to_string(), "    "))
            print("  least coherent:")
            print(textwrap.indent(g.tail(8).round(2).to_string(), "    "))
    return regions


def spread_rows(own: pd.DataFrame, own_col: str, other: pd.DataFrame, other_col: str, share: float) -> list[dict]:
    """Per region of this map: how many regions of the other map hold `share` of its labelled members."""
    assert (own["cid"].to_numpy() == other["cid"].to_numpy()).all(), "the two label files are not aligned"
    rows = []
    for region, group in other.assign(own=own[own_col].to_numpy()).groupby("own"):
        if region == "Unlabelled":
            continue
        there = group[other_col]
        labelled = there[there != "Unlabelled"].value_counts(normalize=True)
        n_for_share = int((labelled.cumsum() < share).sum()) + 1 if len(labelled) else 0
        rows.append(
            {
                "region": region,
                "n": int(len(group)),
                "unlabelled_share": round(float((there == "Unlabelled").mean()), 3),
                "regions_for_share": n_for_share,
                "top_region": labelled.index[0] if len(labelled) else "",
                "top_region_share": round(float(labelled.iloc[0]), 3) if len(labelled) else 0.0,
            }
        )
    return rows


def spread_report(labels: pd.DataFrame, other_files: dict, layout: str, share: float = 0.8) -> list[dict]:
    """How this map's regions (the layer with about 30) scatter across the other map's (the layer with about 60)."""
    if not other_files["labels"].exists():
        print(f"\ncross-map report skipped: {other_files['labels'].name} does not exist yet")
        return []
    other = pd.read_parquet(other_files["labels"])
    own_col, other_col = layer_closest_to(labels, 30), layer_closest_to(other, 60)
    rows = spread_rows(labels, own_col, other, other_col, share)
    df = pd.DataFrame(rows).sort_values(["regions_for_share", "top_region_share"], ascending=[False, True])
    n_own, n_other = labels[own_col].nunique() - 1, other[other_col].nunique() - 1
    print(
        f"\nspread of this map's {n_own} regions ({own_col}) over the {config.other_layout(layout)} map's "
        f"{n_other} regions ({other_col}); regions_for_share = other-map regions holding {share:.0%} of the members"
    )
    print("  most scattered:")
    print(textwrap.indent(df.head(10).to_string(index=False), "    "))
    print("  most concentrated:")
    print(textwrap.indent(df.tail(8).to_string(index=False), "    "))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedding", default=config.EMBED_MODEL_KEY, choices=sorted(config.EMBED_MODELS))
    ap.add_argument("--layout", default="text", choices=sorted(config.LAYOUTS))
    args = ap.parse_args()
    files = config.keyed_files(args.embedding, args.layout)
    complement = config.LAYOUTS[args.layout]["complement"]  # "structure" or "description"
    t0 = time.time()

    corpus, embeddings, coords, labels = load_inputs(files)
    n, k = len(corpus), config.AGREEMENT_K
    print(f"{n} molecules, embeddings {embeddings.shape}, {args.layout} coords {coords.shape}, k = {k}")

    fps = morgan_fingerprints(unescape_smiles(corpus["smiles"]))
    n_bits = fps.sum(1)
    n_distinct = len({row.tobytes() for row in fps})
    print(
        f"Morgan r{config.MORGAN_RADIUS}/{config.MORGAN_N_BITS} fingerprints: {n_distinct} distinct, "
        f"{int((n_bits == 0).sum())} empty, bits set median {int(np.median(n_bits))} ({time.time() - t0:.0f}s)"
    )
    idx_map = knn_layout(coords, k)
    idx_random = np.random.default_rng(config.UMAP_RANDOM_STATE).integers(0, n, (n, k))

    # The complementary space scores the neighbourhoods; the map's own space is found first so the complementary
    # pass can score it too. Stored neighbours are the complementary space's.
    if complement == "structure":
        idx_own, _, _ = cosine_neighbourhoods(embeddings, k)
        print(f"description neighbours found ({time.time() - t0:.0f}s)")
        idx_comp, sims_comp, sim = tanimoto_neighbourhoods(
            fps, k, gather={"map": idx_map, "own": idx_own, "random": idx_random}
        )
        similarity = "Tanimoto similarity of Morgan fingerprints"
        idx_text, idx_morgan = idx_own, idx_comp
    else:
        idx_own, _, _ = tanimoto_neighbourhoods(fps, k)
        print(f"fingerprint neighbours found ({time.time() - t0:.0f}s)")
        idx_comp, sims_comp, sim = cosine_neighbourhoods(
            embeddings, k, gather={"map": idx_map, "own": idx_own, "random": idx_random}
        )
        similarity = "cosine similarity of description embeddings"
        idx_text, idx_morgan = idx_comp, idx_own
    sim["ceiling"] = sims_comp.mean(1)
    print(f"{complement} neighbours found and neighbourhoods scored ({time.time() - t0:.0f}s)")

    floor = float(sim["random"].mean())
    span = np.maximum(sim["ceiling"] - floor, config.COHERENCE_MIN_SPAN)
    coherence = np.clip((sim["map"] - floor) / span, 0, 1)
    shown = config.STRUCTURE_NEIGHBOURS_STORED
    cids = corpus["cid"].to_numpy()
    out = pd.DataFrame(
        {
            "cid": cids,
            "coherence": coherence.astype(np.float32),
            "sim_map": sim["map"],  # complementary similarity of the map neighbours
            "sim_own": sim["own"],  # ... of the neighbours in the map's own high-d space
            "sim_ceiling": sim["ceiling"],  # ... of the complementary space's own neighbours
            "overlap_map_complement": neighbour_overlap(idx_map, idx_comp),
            "overlap_map_own": neighbour_overlap(idx_map, idx_own),
            "overlap_text_morgan": neighbour_overlap(idx_text, idx_morgan),
            "morgan_bits": n_bits.astype(np.int16),
            "neighbour_cids": list(cids[idx_comp[:, :shown]]),
            "neighbour_similarity": list(sims_comp[:, :shown].round(3)),
        }
    )

    # ── Report ──
    print("\nneighbourhood overlap: share of the k nearest neighbours two spaces agree on (random ≈ k/n)")
    overlap_summary = {}
    for col in ("overlap_map_complement", "overlap_map_own", "overlap_text_morgan"):
        v = out[col].to_numpy()
        overlap_summary[col] = describe(v) | {"share_zero": round(float((v == 0).mean()), 3),
                                              "share_half_or_more": round(float((v >= 0.5).mean()), 3)}  # fmt: skip
        print(f"  {col:24s} {fmt(overlap_summary[col])}")
    print(f"\nmean {similarity} of a molecule's k neighbours, by the space that chose them")
    similarity_summary = {name: describe(sim[name]) for name in ("ceiling", "own", "map", "random")}
    for name, d in similarity_summary.items():
        print(f"  {name:8s} {fmt(d)}")
    coherence_summary = describe(coherence)
    print(f"\ncoherence ((map - floor {floor:.3f}) / (ceiling - floor), span floored at {config.COHERENCE_MIN_SPAN}):")
    print(f"  {fmt(coherence_summary)}")
    heavy = corpus["pubchem_heavyAtomCount"].to_numpy(dtype=float)
    by_size = out.groupby(pd.qcut(heavy, 4), observed=True)["coherence"].mean()
    print("coherence by heavy-atom quartile: " + ", ".join(f"{iv} {v:.2f}" for iv, v in by_size.items()))
    regions = region_report(out, labels, heavy)
    spread = spread_report(labels, config.keyed_files(args.embedding, config.other_layout(args.layout)), args.layout)

    # ── Write ──
    validate_stage_output(out, STAGE, ["cid", "coherence", "sim_map", "neighbour_cids"])
    report_delta(STAGE, n, len(out), "nothing dropped")
    write_parquet_safely(out, files["structure"])
    meta = {
        "embedding": args.embedding,
        "layout": args.layout,
        "complement": complement,
        "similarity": similarity,
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
        "random_floor": round(floor, 4),
        "coherence_min_span": config.COHERENCE_MIN_SPAN,
        "neighbours_stored": shown,
        "n_molecules": n,
        "summary": {"overlap": overlap_summary, "similarity": similarity_summary, "coherence": coherence_summary},
        "regions": regions,
        "spread_over_other_map": spread,
        "seconds": round(time.time() - t0),
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_write_text(files["structure_meta"], json.dumps(meta, indent=2))
    print(f"wrote {files['structure']} and {files['structure_meta']} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
