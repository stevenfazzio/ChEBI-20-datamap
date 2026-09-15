"""What stages 04 and 08 share: the Claude namer, the Toponymy clusterer, layer reports and the label files.

Both stages cluster a 2-d layout and hand the clusters to Toponymy for names; they differ in which layout
(descriptions or fingerprints), which text the namer reads and what it is told to name.
"""

import asyncio
import json
import weakref

import numpy as np
import pandas as pd
from toponymy import ToponymyClusterer
from toponymy.llm_wrappers import AsyncLiteLLMNamer

import config
from io_utils import atomic_write_text, write_parquet_safely

PLACEHOLDER = "placeholder (no LLM; --preview)"


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


def make_namer(style: str) -> LoopSafeNamer:
    """What toponymy.llm_wrappers.AsyncAnthropicNamer builds, minus the shared semaphore, plus a style note."""
    return LoopSafeNamer(
        model=f"anthropic/{config.NAMER_MODEL}",
        api_key=config.ANTHROPIC_API_KEY,
        disable_system_prompts=False,
        use_json_object=True,
        llm_specific_instructions=style,
        max_concurrent_requests=config.NAMER_CONCURRENCY,
        provider_kwargs=config.NAMER_PROVIDER_KWARGS,
    )


def fit_clusterer(coords: np.ndarray, embeddings: np.ndarray, min_clusters: int, base_min_cluster_size: int):
    clusterer = ToponymyClusterer(
        min_clusters=min_clusters, base_min_cluster_size=base_min_cluster_size, verbose=False, show_progress_bar=False
    )
    np.random.seed(config.UMAP_RANDOM_STATE)
    # Keyword arguments on purpose: Clusterer.fit takes low-d first, Toponymy.fit takes high-d first.
    clusterer.fit(clusterable_vectors=coords, embedding_vectors=embeddings)
    return clusterer


def describe_layers(clusterer, what: str = "regions") -> list[dict]:
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
            f"  layer {i}: {n:4d} {what}, {rows[-1]['unlabelled_share']:5.1%} unlabelled, "
            f"median size {rows[-1]['median_size']}, largest {rows[-1]['max_size']}"
        )
    return rows


def placeholder_names(clusterer, prefix: str) -> list[list[str]]:
    return [
        [f"{prefix} {i}.{j}" for j in range(int(np.asarray(layer.cluster_labels).max()) + 1)]
        for i, layer in enumerate(clusterer.cluster_layers_)
    ]


def apply_name_overrides(names: list[list[str]], overrides: dict) -> tuple[list[list[str]], list[dict]]:
    """Replace names per `overrides` ({(layer, llm_name): (name, reason)}); returns the new names and what applied."""
    out = [list(layer) for layer in names]
    applied = []
    for (layer, old), (new, reason) in overrides.items():
        if layer < len(out) and old in out[layer]:
            out[layer][out[layer].index(old)] = new
            applied.append({"layer": layer, "from": old, "to": new, "reason": reason})
    return out, applied


def label_frame(corpus: pd.DataFrame, cluster_labels: list[np.ndarray], names: list[list[str]]) -> pd.DataFrame:
    """cid + cluster_layer_i (int, -1 unlabelled) + label_layer_i (name) per layer, finest layer first."""
    assert len(cluster_labels) == len(names)
    labels = pd.DataFrame({"cid": corpus["cid"]})
    for i, (cl, layer_names) in enumerate(zip(cluster_labels, names)):
        cl = np.asarray(cl)
        assert len(cl) == len(corpus) and cl.max() + 1 == len(layer_names)
        labels[f"cluster_layer_{i}"] = cl.astype(np.int32)
        labels[f"label_layer_{i}"] = np.where(
            cl >= 0, np.asarray(layer_names, dtype=object)[np.clip(cl, 0, None)], "Unlabelled"
        )
    n_finest, n_coarsest = labels["label_layer_0"].nunique(), labels[f"label_layer_{len(names) - 1}"].nunique()
    assert n_finest >= n_coarsest, "layers are not finest-first"
    return labels


def write_label_outputs(
    paths: dict, corpus: pd.DataFrame, clusterer, names: list[list[str]], meta: dict, overrides: dict | None = None
) -> None:
    """The label frame, the names per layer, the cluster tree and the run record. `paths` names the four files:
    labels, names, tree, meta. Name overrides, if any, are applied here and listed in the record."""
    names, applied = apply_name_overrides(names, overrides or {})
    meta = meta | {"name_overrides": applied}
    labels = label_frame(corpus, [layer.cluster_labels for layer in clusterer.cluster_layers_], names)

    write_parquet_safely(labels, paths["labels"])
    atomic_write_text(
        paths["names"],
        json.dumps({f"layer_{i}": list(map(str, ln)) for i, ln in enumerate(names)}, indent=1, ensure_ascii=False),
    )
    tree = {f"{k[0]}:{k[1]}": [f"{c[0]}:{c[1]}" for c in v] for k, v in clusterer.cluster_tree_.items()}
    atomic_write_text(paths["tree"], json.dumps(tree, indent=1))
    atomic_write_text(
        paths["meta"], json.dumps(meta | {"layer_order": "finest first (label_layer_0 is the finest)"}, indent=2)
    )
    if applied:
        print("name overrides applied: " + "; ".join(f"{a['from']!r} -> {a['to']!r}" for a in applied))
    print(f"wrote {paths['labels']} with {len(names)} layers; coarsest layer: {list(names[-1])}")
