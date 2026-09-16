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
    out = viz.nearest_line(structure, names, 0.3, show_similarity=True)
    n = viz.clamp  # names are wrapped so CSS can truncate them; the full name stays for search
    assert out[0] == f"{n('Aspirin')} (0.90) · {n('R&D compound')} (0.50)"
    assert n("R&D compound") == "<span>R&amp;D compound</span>"
    assert out[1] == ""  # nothing above the floor: no row at all
    assert out[2] == f"{n('R&D compound')} (0.31) · {n('Toluene')} (0.30)"  # the floor is inclusive
    assert "Toluene" not in out[0]  # STRUCTURE_NEIGHBOURS_SHOWN (2) of the STRUCTURE_NEIGHBOURS_STORED (3)
    # The structure map hides the description cosines (they all read 0.9) and has no floor.
    assert viz.nearest_line(structure, names, 0.0, show_similarity=False)[1] == f"{n('Aspirin')} · {n('R&D compound')}"


def test_build_point_data_adds_extra_lines_only_where_present():
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
    rows = [
        ("Nearest by structure", pd.Series(["X (0.50)", ""])),
        ("Family", None),
        ("Rhea", pd.Series(["", "2 reactions"])),
    ]
    out = viz.build_point_data(corpus, rows)
    a, b = out["body"][0], out["body"][1]
    assert '<div class="hc-k">Nearest by structure</div><div class="hc-v">X (0.50)</div>' in a and "Rhea" not in a
    assert '<div class="hc-k">Rhea</div><div class="hc-v">2 reactions</div>' in b and "Nearest" not in b
    assert "Family" not in a and "Family" not in b  # a missing series contributes no row
    assert "hc-grid" not in viz.build_point_data(corpus)["body"][0]  # no rows, no grid
    # The visible formula carries subscripts; the plain one is hidden but present, so search matches it.
    assert "C<sub>2</sub>H<sub>6</sub>" in a and '<span class="hc-hidden"> · C2H6</span>' in a
    assert '<div class="hc-iupac">ethane</div>' in a and "hc-iupac" not in b
    assert "PubChem CID 2 · ChEBI-20 test split</div>" in b and "hc-hidden" not in b


def test_region_line_escapes_and_skips_unlabelled():
    out = viz.region_line(pd.Series(["Acids & Esters", "Unlabelled", ""]))
    assert out.tolist() == ["Acids &amp; Esters", "", ""]
    assert viz.region_line(np.array(["A", "Unlabelled"])).tolist() == ["A", ""]
    fine, coarse = np.array(["A", "Unlabelled", "Unlabelled"]), np.array(["X", "Y", "Unlabelled"])
    assert viz.region_line(fine, coarse).tolist() == ["A", "Y", ""]  # finest named layer wins


def test_rhea_line_hub_partners_and_more_count():
    rhea = pd.DataFrame(
        {
            "n_reactions": [1398, 12, 1, 0],
            "is_hub": [True, False, False, False],
            "partner_cids": [[], [10, 11, 12, 13, 14], [], []],
        }
    )
    names = {10: "Aspirin", 11: "R&D", 12: "C", 13: "D", 14: "E"}
    out = viz.rhea_line(rhea, names).tolist()
    n = viz.clamp
    assert out[0] == "1,398 reactions (cofactor hub)"
    assert out[1] == f"12 reactions · partners: {n('Aspirin')} · {n('R&D')} · {n('C')} (+2 more)"
    assert out[2] == "1 reaction"
    assert out[3] == ""
