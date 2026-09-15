"""Stage 04: name the regions of the map with Toponymy -> data/labels.parquet (+ topic_names.json,
cluster_tree.json, labels_meta.json).

Clustering happens in the 2-d layout (clusterable_vectors=coords) so named regions correspond to what a
viewer sees; the embeddings supply the semantics for exemplars, keyphrases and the namer.

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

SWEEP_SETTINGS = [  # (min_clusters, base_min_cluster_size)
    (6, 15),
    (6, 20),
    (6, 30),
    (6, 50),
    (4, 20),
]


def load_inputs(files: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    corpus = pd.read_parquet(config.PATHS["corpus"], columns=["cid", "description"])
    emb = np.load(files["npz"], allow_pickle=True)
    lay = np.load(files["umap"], allow_pickle=True)
    cids = corpus["cid"].to_numpy()
    assert (emb["cid"] == cids).all(), "embeddings.npz is not aligned with corpus.parquet"
    assert (lay["cid"] == cids).all(), "umap_coords.npz is not aligned with corpus.parquet"
    return corpus, emb["embeddings"], lay["coords"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedding", default=config.EMBED_MODEL_KEY, choices=sorted(config.EMBED_MODELS))
    ap.add_argument("--sweep", action="store_true", help="report layer sizes at several granularities; no LLM calls")
    ap.add_argument(
        "--preview", action="store_true", help="write placeholder region names for an unnamed map; no LLM calls"
    )
    ap.add_argument("--device", default="auto", help="auto | mps | cuda | cpu | runpod (run this stage on a pod)")
    ap.add_argument("--keep-pod", action="store_true", help="runpod only: leave the pod running afterwards")
    args = ap.parse_args()
    if args.device == "runpod":
        from remote import run_label_on_runpod

        run_label_on_runpod(args.embedding, sweep=args.sweep, keep_pod=args.keep_pod)
        return
    files = config.keyed_files(args.embedding)

    corpus, embeddings, coords = load_inputs(files)
    print(f"{len(corpus)} molecules, embeddings {embeddings.shape}, coords {coords.shape}")

    if args.sweep:
        for min_clusters, base in SWEEP_SETTINGS:
            print(f"\nmin_clusters={min_clusters}, base_min_cluster_size={base}")
            describe_layers(fit_clusterer(coords, embeddings, min_clusters, base))
        return

    clusterer = fit_clusterer(coords, embeddings, config.TOPONYMY_MIN_CLUSTERS, config.TOPONYMY_BASE_MIN_CLUSTER_SIZE)
    print(f"clusterer: min_clusters={config.TOPONYMY_MIN_CLUSTERS}, base size={config.TOPONYMY_BASE_MIN_CLUSTER_SIZE}")
    layer_stats = describe_layers(clusterer)
    if args.preview:
        names = placeholder_names(clusterer, "Region")
        write_outputs(files, corpus, clusterer, names, args.embedding, layer_stats, 0.0, namer_model=PLACEHOLDER)
        return

    assert config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY is not set"
    n_regions = sum(r["n_clusters"] for r in layer_stats)
    print(f"{n_regions} regions to name with {config.NAMER_MODEL} (plus disambiguation passes)")

    namer = make_namer(config.NAMER_STYLE)
    device = None if args.device == "auto" else args.device
    embedder = MoleculeEmbedder(key=args.embedding, device=device)  # keyphrases live in the documents' space
    topic_model = Toponymy(
        llm_wrapper=namer,
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description=config.OBJECT_DESCRIPTION,
        corpus_description=config.CORPUS_DESCRIPTION,
        lowest_detail_level=config.TOPONYMY_LOWEST_DETAIL,
        highest_detail_level=config.TOPONYMY_HIGHEST_DETAIL,
        verbose=True,
    )
    texts = compose_embed_text(corpus).tolist()  # the namer sees exactly the text that was embedded
    t0 = time.time()
    # Keyword arguments on purpose: Toponymy.fit takes high-d first, Clusterer.fit takes low-d first.
    topic_model.fit(objects=texts, embedding_vectors=embeddings, clusterable_vectors=coords)
    elapsed = time.time() - t0
    print(f"Toponymy fit in {elapsed / 60:.1f} min")

    write_outputs(
        files,
        corpus,
        clusterer,
        topic_model.topic_names_,
        args.embedding,
        layer_stats,
        elapsed,
        namer_model=config.NAMER_MODEL,
    )


def write_outputs(files, corpus, clusterer, names, embedding, layer_stats, elapsed, namer_model) -> None:
    meta = {
        "embedding": embedding,
        "namer_model": namer_model,
        "namer_provider_kwargs": config.NAMER_PROVIDER_KWARGS,
        "namer_style": config.NAMER_STYLE,
        "detail_levels": [config.TOPONYMY_LOWEST_DETAIL, config.TOPONYMY_HIGHEST_DETAIL],
        "min_clusters": config.TOPONYMY_MIN_CLUSTERS,
        "base_min_cluster_size": config.TOPONYMY_BASE_MIN_CLUSTER_SIZE,
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
    write_label_outputs(paths, corpus, clusterer, names, meta)


if __name__ == "__main__":
    main()
