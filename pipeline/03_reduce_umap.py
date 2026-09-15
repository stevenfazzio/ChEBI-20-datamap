"""Stage 03: a 2-d layout -> data/umap_coords.npz (+ umap_meta.json), or umap_coords_morgan.npz for the structure map.

`--layout text` (the default) reduces the description embeddings; `--layout morgan` reduces Morgan fingerprints of
the SMILES with the Jaccard metric and otherwise the same neighbourhood settings and seed. Each layout is used for
both its plot and Toponymy's clustering (see CLAUDE.md), so it is computed once with a fixed seed and never
regenerated casually. The fingerprint layout does not depend on the embedding model.
"""

import argparse
import datetime as dt
import json
import time

import numpy as np
import umap

import config
from io_utils import atomic_write_text


def load_vectors(files: dict, layout: str) -> tuple[np.ndarray, object, dict]:
    """cid, the vectors to reduce (dense embeddings, or sparse fingerprint bits) and extra meta for the record."""
    if layout == "text":
        data = np.load(files["npz"], allow_pickle=True)
        return data["cid"], data["embeddings"], {}
    import pandas as pd
    import scipy.sparse

    from structure import morgan_fingerprints, unescape_smiles

    corpus = pd.read_parquet(config.PATHS["corpus"], columns=["cid", "smiles"])
    fps = morgan_fingerprints(unescape_smiles(corpus["smiles"]))
    spec = {"radius": config.MORGAN_RADIUS, "n_bits": config.MORGAN_N_BITS, "chirality": config.MORGAN_CHIRALITY}
    return corpus["cid"].to_numpy(), scipy.sparse.csr_matrix(fps), {"fingerprint": spec}


def park_disconnected(coords: np.ndarray, cids: np.ndarray) -> list[int]:
    """UMAP leaves a vertex at NaN when every neighbour it found sits at the metric's maximum distance (a
    near-empty fingerprint whose rare shared bit the approximate search missed). Nothing is dropped: such points
    are parked just outside the cloud, visibly isolated, where no cluster claims them. Returns their CIDs."""
    disconnected = ~np.isfinite(coords).all(axis=1)
    if not disconnected.any():
        return []
    lo, hi = np.nanmin(coords, axis=0), np.nanmax(coords, axis=0)
    margin = 0.05 * (hi - lo)
    for k, i in enumerate(np.flatnonzero(disconnected)):
        coords[i] = (hi[0] + margin[0], hi[1] + margin[1] - k * margin[1] / 4)
    parked = [int(c) for c in cids[disconnected]]
    print(f"{len(parked)} disconnected vertices parked outside the cloud: cids {parked}")
    return parked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedding", default=config.EMBED_MODEL_KEY, choices=sorted(config.EMBED_MODELS))
    ap.add_argument("--layout", default="text", choices=sorted(config.LAYOUTS))
    args = ap.parse_args()
    files = config.keyed_files(args.embedding, args.layout)

    cid, vectors, extra_meta = load_vectors(files, args.layout)
    print(f"{args.layout} layout from vectors {vectors.shape}")

    params = dict(
        n_neighbors=config.UMAP_N_NEIGHBORS,
        min_dist=config.UMAP_MIN_DIST,
        metric=config.LAYOUTS[args.layout]["metric"],
        n_components=2,
        random_state=config.UMAP_RANDOM_STATE,
    )
    t0 = time.time()
    coords = umap.UMAP(**params, verbose=True).fit_transform(vectors).astype(np.float32)
    elapsed = time.time() - t0
    assert coords.shape == (len(cid), 2)
    parked = park_disconnected(coords, cid)
    assert np.isfinite(coords).all()

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
    meta = (
        params
        | extra_meta
        | {
            "layout": args.layout,
            "embedding": args.embedding if args.layout == "text" else None,
            "umap_learn": umap.__version__,
            "n_documents": int(len(cid)),
            "disconnected_cids": parked,
            "seconds": round(elapsed),
            "reduced_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
    )
    atomic_write_text(files["umap_meta"], json.dumps(meta, indent=2))
    print(f"wrote {files['umap']}")


if __name__ == "__main__":
    main()
