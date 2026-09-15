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
import asyncio
import datetime as dt
import json
import time
import weakref

import numpy as np
import pandas as pd
from toponymy import Toponymy, ToponymyClusterer
from toponymy.llm_wrappers import AsyncLiteLLMNamer

import config
from embedder import MoleculeEmbedder, compose_embed_text
from io_utils import atomic_write_text, write_parquet_safely

PLACEHOLDER = "placeholder (no LLM; --preview)"
SWEEP_SETTINGS = [  # (min_clusters, base_min_cluster_size)
    (6, 15),
    (6, 20),
    (6, 30),
    (6, 50),
    (4, 20),
]


class LoopSafeNamer(AsyncLiteLLMNamer):
    """AsyncLiteLLMNamer with one semaphore per event loop.

    Toponymy 0.5.4 runs every naming pass through its own `asyncio.run()`, while the namer creates a single
    `asyncio.Semaphore` at construction. A semaphore binds to the first loop it has to wait on, so contended
    calls in every later pass fail with "bound to a different event loop" and exhaust their retries. Handing
    out a fresh semaphore per running loop removes the failure.
    """

    def __init__(self, *args, max_concurrent_requests: int = 10, **kwargs):
        self._max_concurrent = max_concurrent_requests
        self._semaphores: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        super().__init__(*args, max_concurrent_requests=max_concurrent_requests, **kwargs)

    @property
    def semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if loop not in self._semaphores:
            self._semaphores[loop] = asyncio.Semaphore(self._max_concurrent)
        return self._semaphores[loop]

    @semaphore.setter
    def semaphore(self, _value) -> None:  # the base __init__ assigns one; per-loop creation replaces it
        pass


def make_namer() -> LoopSafeNamer:
    """What toponymy.llm_wrappers.AsyncAnthropicNamer builds, minus the shared semaphore, plus the style note."""
    return LoopSafeNamer(
        model=f"anthropic/{config.NAMER_MODEL}",
        api_key=config.ANTHROPIC_API_KEY,
        disable_system_prompts=False,
        use_json_object=True,
        llm_specific_instructions=config.NAMER_STYLE,
        max_concurrent_requests=config.NAMER_CONCURRENCY,
        provider_kwargs=config.NAMER_PROVIDER_KWARGS,
    )


def load_inputs(files: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    corpus = pd.read_parquet(config.PATHS["corpus"], columns=["cid", "description"])
    emb = np.load(files["npz"], allow_pickle=True)
    lay = np.load(files["umap"], allow_pickle=True)
    cids = corpus["cid"].to_numpy()
    assert (emb["cid"] == cids).all(), "embeddings.npz is not aligned with corpus.parquet"
    assert (lay["cid"] == cids).all(), "umap_coords.npz is not aligned with corpus.parquet"
    return corpus, emb["embeddings"], lay["coords"]


def fit_clusterer(coords: np.ndarray, embeddings: np.ndarray, min_clusters: int, base_min_cluster_size: int):
    clusterer = ToponymyClusterer(
        min_clusters=min_clusters, base_min_cluster_size=base_min_cluster_size, verbose=False, show_progress_bar=False
    )
    np.random.seed(config.UMAP_RANDOM_STATE)
    # Keyword arguments on purpose: Clusterer.fit takes low-d first, Toponymy.fit takes high-d first.
    clusterer.fit(clusterable_vectors=coords, embedding_vectors=embeddings)
    return clusterer


def describe_layers(clusterer) -> list[dict]:
    rows = []
    for i, layer in enumerate(clusterer.cluster_layers_):
        labels = np.asarray(layer.cluster_labels)
        n = int(labels.max()) + 1
        sizes = np.bincount(labels[labels >= 0])
        rows.append(
            {
                "layer": i,
                "n_clusters": n,
                "unlabelled_share": float((labels < 0).mean()),
                "median_size": int(np.median(sizes)) if n else 0,
                "max_size": int(sizes.max()) if n else 0,
            }
        )
        print(
            f"  layer {i}: {n:4d} regions, {rows[-1]['unlabelled_share']:5.1%} unlabelled, "
            f"median size {rows[-1]['median_size']}, largest {rows[-1]['max_size']}"
        )
    return rows


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
        names = [
            [f"Region {i}.{j}" for j in range(int(np.asarray(layer.cluster_labels).max()) + 1)]
            for i, layer in enumerate(clusterer.cluster_layers_)
        ]
        write_outputs(files, corpus, clusterer, names, args.embedding, layer_stats, 0.0, namer_model=PLACEHOLDER)
        return

    assert config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY is not set"
    n_regions = sum(r["n_clusters"] for r in layer_stats)
    print(f"{n_regions} regions to name with {config.NAMER_MODEL} (plus disambiguation passes)")

    namer = make_namer()
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
    layers = clusterer.cluster_layers_
    assert len(layers) == len(names)  # finest layer first
    labels = pd.DataFrame({"cid": corpus["cid"]})
    for i, (layer, layer_names) in enumerate(zip(layers, names)):
        cl = np.asarray(layer.cluster_labels)
        assert len(cl) == len(corpus) and cl.max() + 1 == len(layer_names)
        labels[f"cluster_layer_{i}"] = cl.astype(np.int32)
        labels[f"label_layer_{i}"] = np.where(
            cl >= 0, np.asarray(layer_names, dtype=object)[np.clip(cl, 0, None)], "Unlabelled"
        )
    n_finest, n_coarsest = labels["label_layer_0"].nunique(), labels[f"label_layer_{len(layers) - 1}"].nunique()
    assert n_finest >= n_coarsest, "layers are not finest-first"

    write_parquet_safely(labels, files["labels"])
    atomic_write_text(
        files["topic_names"],
        json.dumps({f"layer_{i}": list(map(str, ln)) for i, ln in enumerate(names)}, indent=1, ensure_ascii=False),
    )
    tree = {f"{k[0]}:{k[1]}": [f"{c[0]}:{c[1]}" for c in v] for k, v in clusterer.cluster_tree_.items()}
    atomic_write_text(files["cluster_tree"], json.dumps(tree, indent=1))
    meta = {
        "embedding": embedding,
        "namer_model": namer_model,
        "namer_provider_kwargs": config.NAMER_PROVIDER_KWARGS,
        "namer_style": config.NAMER_STYLE,
        "detail_levels": [config.TOPONYMY_LOWEST_DETAIL, config.TOPONYMY_HIGHEST_DETAIL],
        "min_clusters": config.TOPONYMY_MIN_CLUSTERS,
        "base_min_cluster_size": config.TOPONYMY_BASE_MIN_CLUSTER_SIZE,
        "layers": layer_stats,
        "layer_order": "finest first (label_layer_0 is the finest)",
        "minutes": round(elapsed / 60, 1),
        "labelled_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_write_text(files["labels_meta"], json.dumps(meta, indent=2))
    print(f"wrote {files['labels']} with {len(layers)} layers; coarsest layer: {list(names[-1])}")


if __name__ == "__main__":
    main()
