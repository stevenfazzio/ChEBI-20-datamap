"""Stage 08 helpers: reaction parsing, skeleton matching, partner ranking and enzyme-class choice."""

import collections
import gzip
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("rhea", Path(__file__).parents[1] / "pipeline" / "08_rhea.py")
rhea = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rhea)


def test_parse_master_reactions_keeps_only_the_undirected_entry(tmp_path):
    text = (
        "ENTRY       RHEA:10000\nEQUATION    CHEBI:16459 + CHEBI:15377 = CHEBI:31011 + CHEBI:28938\n///\n"
        "ENTRY       RHEA:10001\nEQUATION    CHEBI:16459 + CHEBI:15377 => CHEBI:31011 + CHEBI:28938\n///\n"
        "ENTRY       RHEA:10002\nEQUATION    CHEBI:31011 + CHEBI:28938 <= CHEBI:16459 + CHEBI:15377\n///\n"
        "ENTRY       RHEA:10003\nEQUATION    CHEBI:16459 + CHEBI:15377 <=> CHEBI:31011 + CHEBI:28938\n///\n"
        "ENTRY       RHEA:10004\nEQUATION    2 CHEBI:17484 = CHEBI:16017\n///\n"
    )
    path = tmp_path / "r.txt.gz"
    with gzip.open(path, "wt") as fh:
        fh.write(text)
    got = rhea.parse_master_reactions(path)
    assert got == [
        ("RHEA:10000", ["CHEBI:15377", "CHEBI:16459", "CHEBI:28938", "CHEBI:31011"]),
        ("RHEA:10004", ["CHEBI:16017", "CHEBI:17484"]),
    ]


def test_connectivity_blocks_merge_charge_states_and_flag_unparsable():
    acid, carboxylate, other = "CC(=O)O", "CC(=O)[O-]", "CCO"
    blocks = rhea.connectivity_blocks([acid, carboxylate, other, "not a smiles"])
    assert blocks[0] == blocks[1] and len(blocks[0]) == 14
    assert blocks[2] != blocks[0]
    assert blocks[3] is None


def test_rank_partners_prefers_shared_reactions_then_specific_partners():
    shared = {5: 1, 7: 3, 9: 1}
    degree = {5: 40, 7: 2, 9: 3}
    assert rhea.rank_partners(shared, degree) == [7, 9, 5]


def test_dominant_ec_class_majority_then_lowest_number():
    assert rhea.dominant_ec_class(collections.Counter({"3": 4, "2": 1})) == "Hydrolase"
    assert rhea.dominant_ec_class(collections.Counter({"6": 2, "1": 2})) == "Oxidoreductase"
    assert rhea.dominant_ec_class(collections.Counter()) == "Unassigned"
