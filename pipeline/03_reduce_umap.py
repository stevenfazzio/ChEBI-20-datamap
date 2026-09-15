"""Stage 03: UMAP embeddings -> 2-d -> data/umap_coords.npz (+ umap_meta.json).

This single 2-d layout is used for both the plot and Toponymy's clustering (see CLAUDE.md), so it
is computed once with a fixed seed and never regenerated casually.
"""

import argparse
import datetime as dt
import json
import time

import numpy as np
import umap

import config
from io_utils import atomic_write_text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedding", default=config.EMBED_MODEL_KEY, choices=sorted(config.EMBED_MODELS))
    args = ap.parse_args()
    files = config.keyed_files(args.embedding)

    data = np.load(files["npz"], allow_pickle=True)
    cid, embeddings = data["cid"], data["embeddings"]
    print(f"embeddings {embeddings.shape}")

    params = dict(
        n_neighbors=config.UMAP_N_NEIGHBORS,
        min_dist=config.UMAP_MIN_DIST,
        metric=config.UMAP_METRIC,
        n_components=2,
        random_state=config.UMAP_RANDOM_STATE,
    )
    t0 = time.time()
    coords = umap.UMAP(**params, verbose=True).fit_transform(embeddings).astype(np.float32)
    elapsed = time.time() - t0
    assert coords.shape == (len(cid), 2) and np.isfinite(coords).all()

    # Outlier check: a handful of far points would squeeze the cloud into a corner of the plot.
    lo, hi = np.percentile(coords, [0.1, 99.9], axis=0)
    print(f"UMAP done in {elapsed:.0f}s")
    for axis, name in enumerate("xy"):
        full = f"{coords[:, axis].min():.1f}..{coords[:, axis].max():.1f}"
        print(f"  {name}: range {full}, p0.1..p99.9 {lo[axis]:.1f}..{hi[axis]:.1f}")

    tmp = files["umap"].with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:  # a file handle, or np.savez appends ".npz" to the tmp name
        np.savez(fh, cid=cid, coords=coords)
    assert np.load(tmp, allow_pickle=True)["coords"].shape == coords.shape
    tmp.rename(files["umap"])
    meta = params | {
        "embedding": args.embedding,
        "umap_learn": umap.__version__,
        "n_documents": int(len(cid)),
        "seconds": round(elapsed),
        "reduced_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_write_text(files["umap_meta"], json.dumps(meta, indent=2))
    print(f"wrote {files['umap']}")


if __name__ == "__main__":
    main()
