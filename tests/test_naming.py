"""The label writer shared by stages 04 and 08, and stage 08's namer text."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "pipeline"))

import naming  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "families", Path(__file__).parents[1] / "pipeline" / "08_structure_families.py"
)
families = importlib.util.module_from_spec(spec)
spec.loader.exec_module(families)


def fake_clusterer():
    fine = SimpleNamespace(cluster_labels=np.array([0, 0, 1, 1, 2, -1]))
    coarse = SimpleNamespace(cluster_labels=np.array([0, 0, 0, 0, 1, -1]))
    return SimpleNamespace(cluster_layers_=[fine, coarse], cluster_tree_={(1, 0): [(0, 0), (0, 1)], (1, 1): [(0, 2)]})


def test_write_label_outputs_is_finest_first_and_marks_unlabelled(tmp_path):
    corpus = pd.DataFrame({"cid": [10, 11, 12, 13, 14, 15]})
    paths = {k: tmp_path / f"{k}.out" for k in ("labels", "names", "tree", "meta")}
    paths["labels"] = tmp_path / "labels.parquet"
    names = [["Alkanes", "Alkenes", "Alkynes"], ["Hydrocarbons", "Other"]]
    naming.write_label_outputs(paths, corpus, fake_clusterer(), names, {"note": "test"})
    labels = pd.read_parquet(paths["labels"])
    assert labels["label_layer_0"].tolist() == ["Alkanes", "Alkanes", "Alkenes", "Alkenes", "Alkynes", "Unlabelled"]
    assert labels["label_layer_1"].tolist() == ["Hydrocarbons"] * 4 + ["Other", "Unlabelled"]
    assert labels["cluster_layer_0"].tolist()[-1] == -1
    assert json.loads(paths["names"].read_text()) == {"layer_0": names[0], "layer_1": names[1]}
    assert json.loads(paths["tree"].read_text()) == {"1:0": ["0:0", "0:1"], "1:1": ["0:2"]}
    meta = json.loads(paths["meta"].read_text())
    assert meta["note"] == "test" and meta["layer_order"].startswith("finest first")


def test_placeholder_names_one_per_cluster():
    assert naming.placeholder_names(fake_clusterer(), "Family") == [
        ["Family 0.0", "Family 0.1", "Family 0.2"],
        ["Family 1.0", "Family 1.1"],
    ]


def test_compose_family_text_appends_iupac_when_present():
    corpus = pd.DataFrame(
        {
            "description": ["The molecule is an alkane.", "The molecule is a mystery.  ", "The molecule is a salt."],
            "pubchem_iUPACName": ["ethane", None, ""],
        }
    )
    out = families.compose_family_text(corpus).tolist()
    assert out == [
        "The molecule is an alkane. IUPAC name: ethane.",
        "The molecule is a mystery.",
        "The molecule is a salt.",
    ]


def test_spread_report_counts_regions_to_cover_most_members():
    families_df = pd.DataFrame({"cid": range(10), "label_layer_0": ["Steroids"] * 5 + ["Sugars"] * 4 + ["Unlabelled"]})
    labels_df = pd.DataFrame(
        {
            "cid": range(10),
            # Steroids sit in one map region; Sugars are split evenly over three (one member unlabelled).
            "label_layer_0": ["Hormones"] * 5 + ["A", "B", "C", "Unlabelled", "Z"],
        }
    )
    rows = families.spread_rows(families_df, "label_layer_0", labels_df, "label_layer_0", share=0.8)
    by_name = {r["family"]: r for r in rows}
    assert by_name["Steroids"]["regions_for_share"] == 1 and by_name["Steroids"]["top_region"] == "Hormones"
    assert by_name["Sugars"]["regions_for_share"] == 3  # all three regions of the labelled members for 80% of them
    assert by_name["Sugars"]["unlabelled_share"] == 0.25
    assert "Unlabelled" not in by_name


def test_apply_name_overrides_only_touches_matching_layer_and_name():
    names = [["A", "B"], ["B", "C"]]
    overrides = {(1, "B"): ("Bee", "why"), (0, "Z"): ("Zed", "absent"), (5, "A"): ("Ay", "no such layer")}
    out, applied = naming.apply_name_overrides(names, overrides)
    assert out == [["A", "B"], ["Bee", "C"]] and names == [["A", "B"], ["B", "C"]]  # input untouched
    assert applied == [{"layer": 1, "from": "B", "to": "Bee", "reason": "why"}]
    again, applied_again = naming.apply_name_overrides(out, overrides)
    assert again == out and applied_again == []  # idempotent: the LLM name is gone, so nothing matches
