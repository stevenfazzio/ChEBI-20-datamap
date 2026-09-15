"""The description-derived fields, checked on real ChEBI-20 sentences."""

from extract import derived_fields, organism_of, roles, split_list

AZADIRONE = (
    "The molecule is a tetracyclic triterpenoid that is 4,4,8-trimethylandrosta-1,14-diene substituted by an oxo "
    "group at position 3, an acetoxy group at position 7 and a furan-3-yl group at position 17. Isolated from "
    "Azadirachta indica, it exhibits antiplasmodial and antineoplastic activities. It has a role as an "
    "antineoplastic agent, an antiplasmodial drug and a plant metabolite. It is an acetate ester, a cyclic "
    "terpene ketone, a member of furans, a limonoid and a tetracyclic triterpenoid."
)
NITROSOUREA = (
    "The molecule is a member of the class of N-nitrosoureas that is urea in which one of the nitrogens is "
    "substituted by methyl and nitroso groups. It has a role as a carcinogenic agent, a mutagen, a teratogenic "
    "agent and an alkylating agent."
)
TELLURIUM = (
    "The molecule is the stable isotope of tellurium with relative atomic mass 124.904425, 71.4 atom percent "
    "natural abundance and nuclear spin 1/2."
)
EC_ROLE = (
    "The molecule is a peptide. It has a role as an EC 3.4.21.5 (thrombin) inhibitor, a human urinary metabolite "
    "and an Escherichia coli metabolite. It is a dipeptide and a 1,2-diol."
)


def test_split_list_strips_articles_and_keeps_chemical_commas():
    assert split_list("an acetate ester, a 1,2-diol and the conjugate base") == [
        "acetate ester",
        "1,2-diol",
        "conjugate base",
    ]


def test_roles():
    assert roles(AZADIRONE) == ["antineoplastic agent", "antiplasmodial drug", "plant metabolite"]
    assert roles(NITROSOUREA) == ["carcinogenic agent", "mutagen", "teratogenic agent", "alkylating agent"]
    assert roles(TELLURIUM) == []


def test_role_clause_survives_internal_periods():
    assert roles(EC_ROLE) == [
        "EC 3.4.21.5 (thrombin) inhibitor",
        "human urinary metabolite",
        "Escherichia coli metabolite",
    ]


def test_organism_of():
    assert organism_of("plant metabolite") == "plant"
    assert organism_of("human urinary metabolite") == "human"
    assert organism_of("human xenobiotic metabolite") == "human"
    assert organism_of("Escherichia coli metabolite") == "Escherichia coli"
    assert organism_of("metabolite") == ""
    assert organism_of("fundamental metabolite") == ""
    assert organism_of("drug metabolite") == ""
    assert organism_of("antioxidant") == ""


def test_derived_fields():
    f = derived_fields(AZADIRONE)
    assert f["primary_role"] == "antineoplastic agent"
    assert f["primary_organism"] == "plant"
    f = derived_fields(EC_ROLE)
    assert f["primary_role"] == "EC 3.4.21.5 (thrombin) inhibitor"
    assert f["primary_organism"] == "human"
    f = derived_fields(TELLURIUM)
    assert f == {"roles": [], "primary_role": "", "primary_organism": ""}
