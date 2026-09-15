"""The fingerprint and neighbour-search helpers behind stage 07."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, DataStructs

sys.path.insert(0, str(Path(__file__).parents[1] / "pipeline"))  # stage scripts import siblings the same way

import structure  # noqa: E402

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "OC(=O)c1ccccc1O",  # salicylic acid
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
    "C[C@H](N)C(=O)O",  # L-alanine
    "C[C@@H](N)C(=O)O",  # D-alanine
    "[Na+]",  # a single atom still sets one bit (its radius-0 environment)
]


def test_tanimoto_matches_rdkit_and_excludes_self():
    fps = structure.morgan_fingerprints(SMILES)
    gen = structure.morgan_generator()
    rd = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in SMILES]
    idx, sims, _ = structure.tanimoto_neighbourhoods(fps, k=len(SMILES) - 1, chunk=2)
    for i in range(len(SMILES)):
        assert i not in idx[i]
        assert sorted(idx[i]) == sorted(set(range(len(SMILES))) - {i})
        for j, s in zip(idx[i], sims[i]):
            assert s == pytest.approx(DataStructs.TanimotoSimilarity(rd[i], rd[j]), abs=1e-6)


def test_empty_fingerprints_score_zero_not_nan():
    fps = np.zeros((3, 8), dtype=np.uint8)
    fps[0, :4] = 1
    idx, sims, got = structure.tanimoto_neighbourhoods(fps, k=2, gather={"g": np.array([[1, 2], [0, 2], [0, 1]])})
    assert np.isfinite(sims).all() and sims[1].max() == 0 and sims[2].max() == 0
    assert got["g"].tolist() == [0.0, 0.0, 0.0]


def test_chirality_separates_enantiomers():
    with_chi = structure.morgan_fingerprints(SMILES[3:5])
    without = structure.morgan_fingerprints(SMILES[3:5], structure.morgan_generator(chirality=False))
    assert not np.array_equal(with_chi[0], with_chi[1])
    assert np.array_equal(without[0], without[1])


def test_gather_scores_another_spaces_neighbourhoods():
    fps = structure.morgan_fingerprints(SMILES)
    other = np.array([[1, 2], [0, 2], [0, 1], [4, 0], [3, 0], [0, 1]])
    _, _, got = structure.tanimoto_neighbourhoods(fps, k=2, gather={"other": other}, chunk=4)
    gen = structure.morgan_generator()
    rd = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in SMILES]
    for i, row in enumerate(other):
        expect = np.mean([DataStructs.TanimotoSimilarity(rd[i], rd[j]) for j in row])
        assert got["other"][i] == pytest.approx(expect, abs=1e-6)


def test_unparsable_smiles_is_reported_not_dropped():
    with pytest.raises(ValueError, match="1 SMILES failed"):
        structure.morgan_fingerprints(["CCO", "not a smiles", "CC"])


def test_unescape_smiles():
    s = pd.Series(["C/C=C\\\\C", "CCO"])  # the published form: a doubled backslash
    out = structure.unescape_smiles(s)
    assert out.tolist() == ["C/C=C\\C", "CCO"]
    assert Chem.MolFromSmiles(out[0]) is not None


def test_knn_cosine_exact_and_self_excluded():
    e = np.array([[1, 0], [0.9, 0.1], [0, 1], [-1, 0]], dtype=np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    idx = structure.knn_cosine(e, k=2, chunk=3)
    assert idx[0].tolist() == [1, 2]
    assert idx[3].tolist() == [2, 1]
    assert all(i not in idx[i] for i in range(len(e)))


def test_knn_layout_drops_self_even_with_coincident_points():
    coords = np.array([[0, 0], [0, 0], [0, 0], [1, 0], [5, 5]], dtype=np.float32)
    idx = structure.knn_layout(coords, k=2)
    assert idx.shape == (5, 2)
    assert all(i not in idx[i] for i in range(len(coords)))
    assert set(idx[0]) <= {1, 2, 3}
    assert idx[4][0] == 3 and idx[4][1] in {0, 1, 2}  # the three coincident points tie for second


def test_neighbour_overlap():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[3, 2, 9], [7, 8, 9]])
    assert structure.neighbour_overlap(a, b).tolist() == [2 / 3, 0.0]
