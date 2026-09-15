"""Morgan fingerprints of the SMILES, and exact neighbour search over them and over the other spaces.

Stage 07 uses these to ask how structural the text map's neighbourhoods are. A structure-based layout, if one
is ever built, starts from the same fingerprints.
"""

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.neighbors import NearestNeighbors

import config

CHUNK = 2048  # rows of the n x n similarity held at once: 2048 x 33k float32 is 270 MB


def unescape_smiles(smiles: pd.Series) -> pd.Series:
    """The published files escaped the bond-direction backslash as a doubled one (4,965 rows); undo it."""
    return smiles.str.replace("\\\\", "\\", regex=False)


def morgan_generator(
    radius: int = config.MORGAN_RADIUS, n_bits: int = config.MORGAN_N_BITS, chirality: bool = config.MORGAN_CHIRALITY
):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits, includeChirality=chirality)


def morgan_fingerprints(smiles, generator=None) -> np.ndarray:
    """One uint8 row of bits per SMILES. Every SMILES must parse: a failure is reported, never dropped."""
    RDLogger.DisableLog("rdApp.*")
    generator = generator or morgan_generator()
    smiles = list(smiles)
    out = np.zeros((len(smiles), generator.GetOptions().fpSize), dtype=np.uint8)
    failed = []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            failed.append(i)
            continue
        out[i] = generator.GetFingerprintAsNumPy(mol)
    if failed:
        raise ValueError(
            f"{len(failed)} SMILES failed to parse; first rows {failed[:5]}: {[smiles[i] for i in failed[:3]]}"
        )
    return out


def _top_k(sims: np.ndarray, k: int) -> np.ndarray:
    """Column indices of the k largest entries per row, sorted descending. Self must already be masked."""
    part = np.argpartition(-sims, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(sims, part, 1), axis=1)
    return np.take_along_axis(part, order, 1)


def knn_cosine(embeddings: np.ndarray, k: int, chunk: int = CHUNK) -> np.ndarray:
    """Exact k nearest neighbours by cosine similarity over unit-norm rows, self excluded; (n, k) indices."""
    n = len(embeddings)
    idx = np.empty((n, k), dtype=np.int32)
    for s in range(0, n, chunk):
        sims = embeddings[s : s + chunk] @ embeddings.T
        m = sims.shape[0]
        sims[np.arange(m), np.arange(s, s + m)] = -np.inf
        idx[s : s + m] = _top_k(sims, k)
    return idx


def knn_layout(coords: np.ndarray, k: int) -> np.ndarray:
    """Exact k nearest neighbours in the 2-d layout, self excluded; (n, k) indices.

    Self is dropped wherever it lands in the row: coincident points can push it off position 0.
    """
    n = len(coords)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords, return_distance=False)
    keep = idx != np.arange(n)[:, None]
    keep[~(~keep).any(1), -1] = False  # rows where self never appeared: drop the farthest instead
    return idx[keep].reshape(n, k).astype(np.int32)


def _neighbourhoods(sim_block, n: int, k: int, gather: dict[str, np.ndarray] | None, chunk: int):
    """Exact k nearest neighbours by similarity, self excluded, from a function giving the (m, n) similarity rows
    of a chunk of molecules. Returns the (n, k) neighbour indices, their similarities and, for each array in
    `gather` (name -> (n, m) indices), the mean similarity between every row and the m molecules it names. That
    scores another space's neighbourhoods in the same pass over the n x n similarity instead of a second one."""
    gather = gather or {}
    idx = np.empty((n, k), dtype=np.int32)
    sims_k = np.empty((n, k), dtype=np.float32)
    gathered = {name: np.empty(n, dtype=np.float32) for name in gather}
    for s in range(0, n, chunk):
        m = min(chunk, n - s)
        sims = sim_block(s, m)
        rows = np.arange(m)[:, None]
        for name, ids in gather.items():
            gathered[name][s : s + m] = sims[rows, ids[s : s + m]].mean(1)
        sims[np.arange(m), np.arange(s, s + m)] = -np.inf
        idx[s : s + m] = _top_k(sims, k)
        sims_k[s : s + m] = np.take_along_axis(sims, idx[s : s + m], 1)
    return idx, sims_k, gathered


def tanimoto_neighbourhoods(
    fps: np.ndarray, k: int, gather: dict[str, np.ndarray] | None = None, chunk: int = CHUNK
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Exact k nearest neighbours by Tanimoto similarity over binary fingerprints; see _neighbourhoods."""
    F = fps.astype(np.float32)
    cnt = F.sum(1)

    def block(s, m):
        inter = F[s : s + m] @ F.T
        return inter / np.maximum(cnt[s : s + m, None] + cnt[None, :] - inter, 1)  # empty vs empty scores 0

    return _neighbourhoods(block, len(F), k, gather, chunk)


def cosine_neighbourhoods(
    embeddings: np.ndarray, k: int, gather: dict[str, np.ndarray] | None = None, chunk: int = CHUNK
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Exact k nearest neighbours by cosine similarity over unit-norm rows; see _neighbourhoods."""
    E = embeddings.astype(np.float32)
    return _neighbourhoods(lambda s, m: E[s : s + m] @ E.T, len(E), k, gather, chunk)


def neighbour_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per row, the share of neighbours two (n, k) index arrays agree on."""
    assert a.shape == b.shape, (a.shape, b.shape)
    return (a[:, :, None] == b[:, None, :]).any(2).sum(1) / a.shape[1]
