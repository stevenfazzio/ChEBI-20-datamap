"""Fields read off ChEBI's templated description prose, written against the fetched data (see CLAUDE.md).

ChEBI descriptions follow a house style: "The molecule is a <class> that is ... It has a role as a <role>, a <role>
and a <role>. It is a <class>, a <class> and a <class>." The role clause is present for 54% of ChEBI-20 and is the one
place the text states biological context in a low-cardinality form, so it feeds the two derived colormaps:
`primary_role` (the first role listed) and `primary_organism` (the organism of the first "<organism> metabolite"
role). Everything else the prose states (classes, source organism, conjugates) stays in the description itself,
which is on the hovercard and in search.
"""

import re

ROLE_RE = re.compile(r"\bhas a role as (.+?)\.(?=\s|$)")
METABOLITE_RE = re.compile(r"^(.+?) metabolite$")

# Trailing qualifiers ChEBI attaches to an organism in metabolite roles ("human urinary metabolite").
ORGANISM_QUALIFIERS = ("xenobiotic", "urinary", "blood serum", "saliva", "faecal", "fecal", "milk", "sweat", "gut")
# "<word> metabolite" roles whose first word is not an organism.
NOT_ORGANISMS = {"", "fundamental", "drug", "xenobiotic", "secondary", "primary", "volatile", "cofactor"}


def split_list(clause: str) -> list[str]:
    """'a metabolite, an antioxidant and a plant metabolite' -> ['metabolite', 'antioxidant', 'plant metabolite'].

    Splits on ', ' and ' and ' only; a bare comma inside a chemical name ('1,2-diol') is left alone.
    """
    out = []
    for part in re.split(r",\s+|\s+and\s+", clause):
        part = re.sub(r"^(?:an?|the)\s+", "", part.strip().rstrip("."))
        if part:
            out.append(part)
    return out


def roles(description: str) -> list[str]:
    m = ROLE_RE.search(description)
    return split_list(m.group(1)) if m else []


def organism_of(role: str) -> str:
    """'human urinary metabolite' -> 'human'; 'Escherichia coli metabolite' -> 'Escherichia coli'; else ''."""
    m = METABOLITE_RE.match(role)
    if not m:
        return ""
    organism = m.group(1).strip()
    for q in ORGANISM_QUALIFIERS:
        if organism.endswith(" " + q):
            organism = organism[: -len(q) - 1].strip()
    return "" if organism in NOT_ORGANISMS else organism


def derived_fields(description: str) -> dict:
    rs = roles(description)
    organisms = [o for o in (organism_of(r) for r in rs) if o]
    return {
        "roles": rs,
        "primary_role": rs[0] if rs else "",
        "primary_organism": organisms[0] if organisms else "",
    }
