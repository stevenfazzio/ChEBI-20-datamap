"""Stage 08: structural families -> data/families.parquet (+ family_names.json, family_tree.json, families_meta.json).

Groups the molecules by chemical structure rather than by description. A UMAP of the Morgan fingerprints
(Jaccard metric, otherwise the text layout's settings; cached as data/umap_coords_morgan.npz) is clustered with
Toponymy's clusterer, and Claude names each family from its members' descriptions and IUPAC names with an
instruction to name the shared structure, not roles or sources. The text embeddings supply exemplars and
keyphrases as in stage 04. Stage 05 colours the description map by family; the fingerprint layout is also the
starting point for a structure-based map, if one is ever built.

`--sweep` reports layer sizes at several granularities and `--preview` writes placeholder names; neither calls
the LLM. `--device runpod` runs the naming on a pod as stage 04 does; the layout must already exist locally.
"""

import argparse
import datetime as dt
import json
import time

import numpy as np
import pandas as pd
from toponymy import Toponymy

import config
from embedder import MoleculeEmbedder
from io_utils import atomic_write_text
from naming import PLACEHOLDER, describe_layers, fit_clusterer, make_namer, placeholder_names, write_label_outputs

SWEEP_SETTINGS = [  # (min_clusters, base_min_cluster_size)
    (6, 50),
    (6, 100),
    (6, 150),
    (6, 200),
    (6, 300),
]


def compose_family_text(corpus: pd.DataFrame) -> pd.Series:
    """What the namer reads: the description, then the IUPAC name, which is structure written in words."""
    iupac = corpus["pubchem_iUPACName"]
    has = iupac.notna() & (iupac.astype(str).str.strip() != "")
    return corpus["description"].str.strip() + np.where(has, " IUPAC name: " + iupac.astype(str).str.strip() + ".", "")


def layout_params() -> dict:
    return dict(
        n_neighbors=config.UMAP_N_NEIGHBORS,
        min_dist=config.UMAP_MIN_DIST,
        metric=config.STRUCTURE_UMAP_METRIC,
        n_components=2,
        random_state=config.UMAP_RANDOM_STATE,
    )


def fingerprint_spec() -> dict:
    return {"radius": config.MORGAN_RADIUS, "n_bits": config.MORGAN_N_BITS, "chirality": config.MORGAN_CHIRALITY}


def fingerprint_layout(corpus: pd.DataFrame) -> np.ndarray:
    """UMAP of the Morgan fingerprints, computed once and cached; a cache made with other settings is refused."""
    path, meta_path = config.PATHS["structure_umap"], config.PATHS["structure_umap_meta"]
    cids = corpus["cid"].to_numpy()
    params = layout_params()
    if path.exists():
        data = np.load(path, allow_pickle=True)
        assert (data["cid"] == cids).all(), f"{path} is not aligned with corpus.parquet"
        meta = json.loads(meta_path.read_text())
        stale = {
            k: (meta.get(k), v) for k, v in (params | {"fingerprint": fingerprint_spec()}).items() if meta.get(k) != v
        }
        assert not stale, f"{path} was made with other settings {stale}: delete it to recompute"
        print(f"fingerprint layout loaded from {path}")
        return data["coords"]

    import scipy.sparse
    import umap

    from structure import morgan_fingerprints, unescape_smiles

    t0 = time.time()
    fps = morgan_fingerprints(unescape_smiles(corpus["smiles"]))
    print(f"fingerprints {fps.shape} in {time.time() - t0:.0f}s; UMAP ({params['metric']}) ...", flush=True)
    coords = umap.UMAP(**params, verbose=True).fit_transform(scipy.sparse.csr_matrix(fps)).astype(np.float32)
    elapsed = time.time() - t0
    assert coords.shape == (len(cids), 2)
    # A molecule whose approximate neighbours all sit at Jaccard distance 1 (a near-empty fingerprint whose rare
    # shared bit the neighbour search missed) is disconnected by UMAP and left at NaN. Nothing is dropped: such
    # points are parked just outside the cloud, visibly isolated, where no cluster claims them.
    disconnected = ~np.isfinite(coords).all(axis=1)
    if disconnected.any():
        lo, hi = np.nanmin(coords, axis=0), np.nanmax(coords, axis=0)
        margin = 0.05 * (hi - lo)
        for k, i in enumerate(np.flatnonzero(disconnected)):
            coords[i] = (hi[0] + margin[0], hi[1] + margin[1] - k * margin[1] / 4)
        parked = cids[disconnected].tolist()
        print(f"{len(parked)} disconnected vertices parked outside the cloud: cids {parked}")
    assert np.isfinite(coords).all()
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:  # a file handle, or np.savez appends ".npz" to the tmp name
        np.savez(fh, cid=cids, coords=coords)
    assert np.load(tmp, allow_pickle=True)["coords"].shape == coords.shape
    tmp.rename(path)
    meta = params | {
        "fingerprint": fingerprint_spec(),
        "umap_learn": umap.__version__,
        "n_documents": int(len(cids)),
        "disconnected_cids": [int(c) for c in cids[disconnected]],
        "seconds": round(elapsed),
        "reduced_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_write_text(meta_path, json.dumps(meta, indent=2))
    print(f"UMAP done in {elapsed:.0f}s; wrote {path}")
    return coords


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedding", default=config.EMBED_MODEL_KEY, choices=sorted(config.EMBED_MODELS))
    ap.add_argument("--sweep", action="store_true", help="report layer sizes at several granularities; no LLM calls")
    ap.add_argument("--preview", action="store_true", help="write placeholder family names; no LLM calls")
    ap.add_argument("--device", default="auto", help="auto | mps | cuda | cpu | runpod (name on a pod)")
    ap.add_argument("--keep-pod", action="store_true", help="runpod only: leave the pod running afterwards")
    args = ap.parse_args()
    if args.device == "runpod":
        from remote import run_families_on_runpod

        run_families_on_runpod(args.embedding, keep_pod=args.keep_pod)
        report_spread(config.keyed_files(args.embedding))
        return
    files = config.keyed_files(args.embedding)

    corpus = pd.read_parquet(config.PATHS["corpus"], columns=["cid", "smiles", "description", "pubchem_iUPACName"])
    cids = corpus["cid"].to_numpy()
    emb = np.load(files["npz"], allow_pickle=True)
    assert (emb["cid"] == cids).all(), "embeddings.npz is not aligned with corpus.parquet"
    embeddings = emb["embeddings"]
    coords = fingerprint_layout(corpus)
    print(f"{len(corpus)} molecules, embeddings {embeddings.shape}, fingerprint layout {coords.shape}")

    if args.sweep:
        for min_clusters, base in SWEEP_SETTINGS:
            print(f"\nmin_clusters={min_clusters}, base_min_cluster_size={base}")
            describe_layers(fit_clusterer(coords, embeddings, min_clusters, base), "families")
        return

    clusterer = fit_clusterer(coords, embeddings, config.FAMILY_MIN_CLUSTERS, config.FAMILY_BASE_MIN_CLUSTER_SIZE)
    print(f"clusterer: min_clusters={config.FAMILY_MIN_CLUSTERS}, base size={config.FAMILY_BASE_MIN_CLUSTER_SIZE}")
    layer_stats = describe_layers(clusterer, "families")
    if args.preview:
        write_outputs(
            files,
            corpus,
            clusterer,
            placeholder_names(clusterer, "Family"),
            args.embedding,
            layer_stats,
            0.0,
            PLACEHOLDER,
        )
        return

    assert config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY is not set"
    n_families = sum(r["n_clusters"] for r in layer_stats)
    print(f"{n_families} families to name with {config.NAMER_MODEL} (plus disambiguation passes)")
    device = None if args.device == "auto" else args.device
    embedder = MoleculeEmbedder(key=args.embedding, device=device)  # keyphrases live in the documents' space
    topic_model = Toponymy(
        llm_wrapper=make_namer(config.FAMILY_NAMER_STYLE),
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description=config.FAMILY_OBJECT_DESCRIPTION,
        corpus_description=config.CORPUS_DESCRIPTION,
        lowest_detail_level=config.FAMILY_LOWEST_DETAIL,
        highest_detail_level=config.FAMILY_HIGHEST_DETAIL,
        verbose=True,
    )
    texts = compose_family_text(corpus).tolist()
    t0 = time.time()
    # Keyword arguments on purpose: Toponymy.fit takes high-d first, Clusterer.fit takes low-d first.
    topic_model.fit(objects=texts, embedding_vectors=embeddings, clusterable_vectors=coords)
    elapsed = time.time() - t0
    print(f"Toponymy fit in {elapsed / 60:.1f} min")
    write_outputs(
        files, corpus, clusterer, topic_model.topic_names_, args.embedding, layer_stats, elapsed, config.NAMER_MODEL
    )
    for i, layer_names in reversed(list(enumerate(topic_model.topic_names_))):
        if len(layer_names) <= 60:
            print(f"\nlayer {i} ({len(layer_names)} families):")
            for name in layer_names:
                print(f"  {name}")
    report_spread(files)


def layer_closest_to(df: pd.DataFrame, target: int) -> str:
    """The label_layer_i column whose number of named groups is closest to `target`."""
    cols = [c for c in df.columns if c.startswith("label_layer_")]
    return min(cols, key=lambda c: abs(df[c].nunique() - target))


def spread_rows(families: pd.DataFrame, fam_col: str, labels: pd.DataFrame, reg_col: str, share: float) -> list[dict]:
    """Per family: how many description-map regions hold `share` of its labelled members, and the largest one."""
    assert (families["cid"].to_numpy() == labels["cid"].to_numpy()).all(), "families and labels are not aligned"
    rows = []
    for family, group in labels.assign(family=families[fam_col].to_numpy()).groupby("family"):
        if family == "Unlabelled":
            continue
        regions = group[reg_col]
        labelled = regions[regions != "Unlabelled"].value_counts(normalize=True)
        n_for_share = int((labelled.cumsum() < share).sum()) + 1 if len(labelled) else 0
        rows.append(
            {
                "family": family,
                "n": int(len(group)),
                "unlabelled_share": round(float((regions == "Unlabelled").mean()), 3),
                "regions_for_share": n_for_share,
                "top_region": labelled.index[0] if len(labelled) else "",
                "top_region_share": round(float(labelled.iloc[0]), 3) if len(labelled) else 0.0,
            }
        )
    return rows


def report_spread(files: dict, share: float = 0.8) -> None:
    """How the families scatter across the description map. Needs the map's labels, so it runs locally only."""
    if not (files["labels"].exists() and files["families"].exists()):
        print("spread report skipped: labels.parquet or families.parquet missing")
        return
    families, labels = pd.read_parquet(files["families"]), pd.read_parquet(files["labels"])
    fam_col, reg_col = layer_closest_to(families, 30), layer_closest_to(labels, 60)
    rows = spread_rows(families, fam_col, labels, reg_col, share)
    df = pd.DataFrame(rows).sort_values(["regions_for_share", "top_region_share"], ascending=[False, True])
    n_fam, n_reg = families[fam_col].nunique() - 1, labels[reg_col].nunique() - 1
    print(f"\nspread of the {n_fam} families ({fam_col}) over the map's {n_reg} regions ({reg_col}):")
    print(f"  regions_for_share = map regions holding {share:.0%} of the family's labelled members")
    print("  most scattered:")
    print(df.head(10).to_string(index=False))
    print("  most concentrated:")
    print(df.tail(8).to_string(index=False))


def write_outputs(files, corpus, clusterer, names, embedding, layer_stats, elapsed, namer_model) -> None:
    meta = {
        "embedding": embedding,
        "layout": config.PATHS["structure_umap"].name,
        "namer_model": namer_model,
        "namer_provider_kwargs": config.NAMER_PROVIDER_KWARGS,
        "namer_style": config.FAMILY_NAMER_STYLE,
        "object_description": config.FAMILY_OBJECT_DESCRIPTION,
        "detail_levels": [config.FAMILY_LOWEST_DETAIL, config.FAMILY_HIGHEST_DETAIL],
        "min_clusters": config.FAMILY_MIN_CLUSTERS,
        "base_min_cluster_size": config.FAMILY_BASE_MIN_CLUSTER_SIZE,
        "layers": layer_stats,
        "minutes": round(elapsed / 60, 1),
        "labelled_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    paths = {
        "labels": files["families"],
        "names": files["family_names"],
        "tree": files["family_tree"],
        "meta": files["families_meta"],
    }
    write_label_outputs(paths, corpus, clusterer, names, meta, config.FAMILY_NAME_OVERRIDES)


if __name__ == "__main__":
    main()
