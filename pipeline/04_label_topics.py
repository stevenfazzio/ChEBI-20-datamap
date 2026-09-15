"""Stage 04: name the regions of a map with Toponymy -> data/labels.parquet (+ topic_names.json, cluster_tree.json,
labels_meta.json), or the `_morgan` versions for the structure map.

Clustering happens in the 2-d layout (clusterable_vectors=coords) so named regions correspond to what a viewer
sees; the description embeddings supply the semantics for exemplars, keyphrases and the namer on both maps. For
the structure map (`--layout morgan`) the namer also sees each molecule's IUPAC name and is told to name the
shared chemical structure rather than roles or sources, since those regions are structural clusters; hand
corrections to its names live in config.STRUCTURE_NAME_OVERRIDES and are recorded in the meta.

`--sweep` fits the clusterer at several granularities and reports layer sizes without calling the LLM.
`--preview` fits the configured granularity and writes placeholder names ("Region 2.17") so stage 05 can
render an unnamed map for checking the layout and hovercard; the real run overwrites those files.
Layer 0 is the FINEST layer, both in memory and on disk (label_layer_0 = finest), matching Toponymy
and the order DataMapPlot expects.
"""

import argparse
import datetime as dt
import time

import numpy as np
import pandas as pd
from toponymy import Toponymy

import config
from embedder import MoleculeEmbedder, compose_embed_text
from naming import PLACEHOLDER, describe_layers, fit_clusterer, make_namer, placeholder_names, write_label_outputs


def compose_namer_text(corpus: pd.DataFrame, layout: str) -> pd.Series:
    """What the namer reads. The text map's namer sees exactly the text that was embedded; the structure map's
    namer also gets the IUPAC name, which is structure written in words."""
    text = compose_embed_text(corpus)
    if not config.LAYOUTS[layout]["namer_sees_iupac"]:
        return text
    iupac = corpus["pubchem_iUPACName"]
    has = iupac.notna() & (iupac.astype(str).str.strip() != "")
    return text + np.where(has, " IUPAC name: " + iupac.astype(str).str.strip() + ".", "")


def load_inputs(files: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    corpus = pd.read_parquet(config.PATHS["corpus"], columns=["cid", "description", "pubchem_iUPACName"])
    emb = np.load(files["npz"], allow_pickle=True)
    lay = np.load(files["umap"], allow_pickle=True)
    cids = corpus["cid"].to_numpy()
    assert (emb["cid"] == cids).all(), "embeddings.npz is not aligned with corpus.parquet"
    assert (lay["cid"] == cids).all(), f"{files['umap'].name} is not aligned with corpus.parquet"
    return corpus, emb["embeddings"], lay["coords"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedding", default=config.EMBED_MODEL_KEY, choices=sorted(config.EMBED_MODELS))
    ap.add_argument("--layout", default="text", choices=sorted(config.LAYOUTS))
    ap.add_argument("--sweep", action="store_true", help="report layer sizes at several granularities; no LLM calls")
    ap.add_argument(
        "--preview", action="store_true", help="write placeholder region names for an unnamed map; no LLM calls"
    )
    ap.add_argument("--device", default="auto", help="auto | mps | cuda | cpu | runpod (run this stage on a pod)")
    ap.add_argument("--keep-pod", action="store_true", help="runpod only: leave the pod running afterwards")
    args = ap.parse_args()
    if args.device == "runpod":
        from remote import run_label_on_runpod

        run_label_on_runpod(args.embedding, args.layout, sweep=args.sweep, keep_pod=args.keep_pod)
        return
    files = config.keyed_files(args.embedding, args.layout)
    spec = config.LAYOUTS[args.layout]

    corpus, embeddings, coords = load_inputs(files)
    print(f"{len(corpus)} molecules, embeddings {embeddings.shape}, {args.layout} coords {coords.shape}")

    if args.sweep:
        for min_clusters, base in spec["sweep"]:
            print(f"\nmin_clusters={min_clusters}, base_min_cluster_size={base}")
            describe_layers(fit_clusterer(coords, embeddings, min_clusters, base))
        return

    clusterer = fit_clusterer(coords, embeddings, spec["min_clusters"], spec["base_min_cluster_size"])
    print(f"clusterer: min_clusters={spec['min_clusters']}, base size={spec['base_min_cluster_size']}")
    layer_stats = describe_layers(clusterer)
    if args.preview:
        names = placeholder_names(clusterer, "Region")
        write_outputs(files, corpus, clusterer, names, args, layer_stats, 0.0, namer_model=PLACEHOLDER)
        return

    assert config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY is not set"
    n_regions = sum(r["n_clusters"] for r in layer_stats)
    print(f"{n_regions} regions to name with {config.NAMER_MODEL} (plus disambiguation passes)")

    namer = make_namer(spec["namer_style"])
    device = None if args.device == "auto" else args.device
    embedder = MoleculeEmbedder(key=args.embedding, device=device)  # keyphrases live in the documents' space
    topic_model = Toponymy(
        llm_wrapper=namer,
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description=spec["object_description"],
        corpus_description=config.CORPUS_DESCRIPTION,
        lowest_detail_level=spec["detail_levels"][0],
        highest_detail_level=spec["detail_levels"][1],
        verbose=True,
    )
    texts = compose_namer_text(corpus, args.layout).tolist()
    t0 = time.time()
    # Keyword arguments on purpose: Toponymy.fit takes high-d first, Clusterer.fit takes low-d first.
    topic_model.fit(objects=texts, embedding_vectors=embeddings, clusterable_vectors=coords)
    elapsed = time.time() - t0
    print(f"Toponymy fit in {elapsed / 60:.1f} min")

    write_outputs(files, corpus, clusterer, topic_model.topic_names_, args, layer_stats, elapsed, config.NAMER_MODEL)


def write_outputs(files, corpus, clusterer, names, args, layer_stats, elapsed, namer_model) -> None:
    spec = config.LAYOUTS[args.layout]
    meta = {
        "embedding": args.embedding,
        "layout": args.layout,
        "namer_model": namer_model,
        "namer_provider_kwargs": config.NAMER_PROVIDER_KWARGS,
        "namer_style": spec["namer_style"],
        "object_description": spec["object_description"],
        "namer_sees_iupac": spec["namer_sees_iupac"],
        "detail_levels": list(spec["detail_levels"]),
        "min_clusters": spec["min_clusters"],
        "base_min_cluster_size": spec["base_min_cluster_size"],
        "layers": layer_stats,
        "minutes": round(elapsed / 60, 1),
        "labelled_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    paths = {
        "labels": files["labels"],
        "names": files["topic_names"],
        "tree": files["cluster_tree"],
        "meta": files["labels_meta"],
    }
    write_label_outputs(paths, corpus, clusterer, names, meta, spec["name_overrides"])


if __name__ == "__main__":
    main()
