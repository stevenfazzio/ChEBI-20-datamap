"""Helpers of stage 05 (loaded by path because the stage's file name starts with a digit)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("visualize", Path(__file__).parents[1] / "pipeline" / "05_visualize.py")
viz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viz)


def test_formula_html_subscripts_and_charge():
    assert viz.formula_html("C11H8N2S") == "C<sub>11</sub>H<sub>8</sub>N<sub>2</sub>S"
    assert viz.formula_html("C20H31O4-") == "C<sub>20</sub>H<sub>31</sub>O<sub>4</sub><sup>−</sup>"
    assert viz.formula_html("C5H14NO+") == "C<sub>5</sub>H<sub>14</sub>NO<sup>+</sup>"
    assert viz.formula_html("C6H4O7-3") == "C<sub>6</sub>H<sub>4</sub>O<sub>7</sub><sup>−3</sup>"
    assert viz.formula_html(None) == ""
    assert viz.formula_html(float("nan")) == ""


def test_charge_bucket():
    assert [viz.charge_bucket(c) for c in (-4, -2, -1, 0, 1, 2, 5)] == [
        "−3 or lower",
        "−2",
        "−1",
        "0",
        "+1",
        "+2",
        "+3 or higher",
    ]
    assert viz.charge_bucket(float("nan")) == "Unknown"


def test_top_n_pools_the_tail_and_labels_empties():
    values = pd.Series(["plant"] * 5 + ["human"] * 3 + ["rat"] * 1 + ["mouse"] * 1 + [""] * 2)
    out = viz.top_n(values, 2, "Other organism", "None stated")
    assert list(out[:5]) == ["plant"] * 5
    assert list(out[5:8]) == ["human"] * 3
    assert list(out[8:10]) == ["Other organism"] * 2
    assert list(out[10:]) == ["None stated"] * 2


def test_categorical_pins_neutral_to_grey_and_covers_every_category():
    values = np.array(["a", "b", "None stated", "c", "a"])
    meta, vals = viz.categorical("f", "F", values, neutral="None stated")
    assert meta["kind"] == "categorical" and set(meta["color_mapping"]) == {"a", "b", "c", "None stated"}
    assert meta["color_mapping"]["None stated"] == viz.NEUTRAL
    assert len({meta["color_mapping"][k] for k in ("a", "b", "c")}) == 3
    assert (vals == values).all()


def test_nearest_line_drops_weak_neighbours_and_escapes():
    structure = pd.DataFrame(
        {
            "neighbour_cids": [[10, 11, 12], [10, 11, 12], [11, 12, 10]],
            "neighbour_similarity": [[0.9, 0.5, 0.2], [0.29, 0.1, 0.05], [0.31, 0.3, 0.29]],
        }
    )
    names = {10: "Aspirin", 11: "R&D compound", 12: "Toluene"}
    out = viz.nearest_line(structure, names, "Nearest by structure", 0.3)
    assert out[0] == "Nearest by structure: Aspirin (0.90) · R&amp;D compound (0.50)"
    assert out[1] == ""  # nothing above the floor: no line at all
    assert out[2] == "Nearest by structure: R&amp;D compound (0.31) · Toluene (0.30)"  # the floor is inclusive
    assert "Toluene" not in out[0]  # STRUCTURE_NEIGHBOURS_SHOWN (2) of the STRUCTURE_NEIGHBOURS_STORED (3)
    assert viz.nearest_line(structure, names, "Nearest by description", 0.0)[1].startswith(
        "Nearest by description: Aspirin"
    )


def test_build_point_data_includes_the_nearest_line_only_when_present():
    corpus = pd.DataFrame(
        {
            "cid": [1, 2],
            "name": ["A", "B"],
            "split": ["train", "test"],
            "description": ["The molecule is a thing.", "The molecule is another."],
            "pubchem_molecularFormula": ["C2H6", None],
            "pubchem_molecularWeight": [30.07, None],
            "pubchem_charge": [0, None],
            "pubchem_xLogP": [1.0, None],
            "pubchem_iUPACName": ["ethane", None],
        }
    )
    out = viz.build_point_data(corpus, pd.Series(["Nearest by structure: X (0.50)", ""]))
    assert "Nearest by structure: X (0.50)" in out["body"][0]
    assert "Nearest by structure" not in out["body"][1]
    assert "Nearest by structure" not in viz.build_point_data(corpus)["body"][0]  # the line is optional


def test_build_point_data_cross_line_is_optional_and_escaped():
    corpus = pd.DataFrame(
        {
            "cid": [1, 2],
            "name": ["A", "B"],
            "split": ["train", "test"],
            "description": ["The molecule is a thing.", "The molecule is another."],
            "pubchem_molecularFormula": ["C2H6", None],
            "pubchem_molecularWeight": [30.07, None],
            "pubchem_charge": [0, None],
            "pubchem_xLogP": [1.0, None],
            "pubchem_iUPACName": ["ethane", None],
        }
    )
    out = viz.build_point_data(
        corpus, cross=pd.Series(["Acids & Esters", "Unlabelled"]), cross_label="Structural family"
    )
    assert "Structural family: Acids &amp; Esters" in out["body"][0]
    assert "Structural family" not in out["body"][1]  # Unlabelled shows nothing
