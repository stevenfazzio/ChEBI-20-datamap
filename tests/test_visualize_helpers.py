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
