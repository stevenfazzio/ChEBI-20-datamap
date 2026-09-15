# ChEBI-20 Molecule Map

An interactive map of the [ChEBI-20](https://github.com/blender-nlp/MolT5) dataset: 33,008 molecules, each
placed by the meaning of its ChEBI description, with regions named at four zoom levels.

**Live map:** https://stevenfazzio.com/ChEBI-20-datamap/

## What the map shows

ChEBI-20 (Edwards et al. 2022) pairs a PubChem compound ID and SMILES string with the natural-language
description ChEBI curators wrote for the molecule: its chemical class, how it relates to other compounds,
its biological roles, and where it was isolated. This map embeds those descriptions and lays them out in
two dimensions, so molecules that ChEBI describes similarly sit near each other. Hover for the compound
name, formula, weight, charge, XLogP, structure drawing, full description, structural family and the two
molecules nearest to it by chemical structure; click to open the PubChem page; search by name, formula, CID,
family or any phrase in the description. The colour menu switches between the region colouring and eight
metadata views: metabolite organism, first-stated biological role, structural family, formal charge,
molecular weight, XLogP, structural coherence and dataset split. The last two, and the families, are
explained below.

## How it is built

| Stage | Script | What it does |
|---|---|---|
| 00 | `pipeline/00_fetch.py` | Downloads the three MolT5 split files and merges them (train 26,407 / validation 3,301 / test 3,300). |
| 01 | `pipeline/01_enrich.py` | Fetches PubChem properties for every CID in batches; reads role and organism fields off the description prose. |
| 02 | `pipeline/02_embed.py` | Embeds each description with [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B), locally or on a Runpod GPU. |
| 03 | `pipeline/03_reduce_umap.py` | UMAP to two dimensions with a fixed seed. |
| 04 | `pipeline/04_label_topics.py` | [Toponymy](https://github.com/TutteInstitute/toponymy) clusters the 2-d layout and names the regions with Claude. |
| 07 | `pipeline/07_structure_agreement.py` | Morgan fingerprints of every SMILES ([RDKit](https://www.rdkit.org/)); scores how structurally similar each molecule's map neighbours are and finds its nearest structural neighbours. |
| 08 | `pipeline/08_structure_families.py` | UMAP of the fingerprints, clustered with Toponymy into structural families that Claude names for their shared structure. |
| 05 | `pipeline/05_visualize.py` | [DataMapPlot](https://github.com/TutteInstitute/datamapplot) renders the map into `docs/`. |

```bash
uv sync --extra dev
make fetch enrich embed umap label structure families visualize   # or: make map (stages 02-08)
make serve                                     # http://127.0.0.1:8765/
```

Stages 04 and 08 call the Anthropic API and are the only stages that cost money. `make label` and
`make families` run them with the environment variables they need on macOS. Region and family names are
LLM-generated labels for clusters; they summarise what a group's molecules have in common and are not ChEBI
classifications.

## How structural is the map?

The layout comes from the descriptions alone, so it is fair to ask how far it agrees with chemical structure.
Stage 07 answers that with Morgan fingerprints (radius 2, 2,048 bits, chirality on) of every SMILES, comparing
each molecule's 15 nearest neighbours on the map, in the text-embedding space and in fingerprint space.

| Neighbourhoods compared (15 nearest) | Neighbours shared, mean | Median | No neighbour shared |
|---|---|---|---|
| Text embedding vs fingerprints | 0.20 | 0.13 | 19% |
| Map layout vs fingerprints | 0.16 | 0.07 | 33% |
| Text embedding vs map layout | 0.42 | 0.40 | 3% |

| A molecule's 15 neighbours, chosen by | Mean Tanimoto similarity to it |
|---|---|
| Fingerprints (the ceiling) | 0.58 |
| Text embedding | 0.38 |
| Map layout | 0.34 |
| Random molecules (the floor) | 0.09 |

So the description map is about half structural: its neighbours are four times more similar than chance, but
only one in five is the molecule fingerprints would choose. Agreement rises with molecule size and does not
depend on description length or on whether a biological role is stated. Where ChEBI's prose is effectively a
structure written in words (lipids: "a phosphatidylserine 34:2 in which the acyl groups ..."), the map is
structural; where the prose groups by use or source, it is not. Dodemorph, for example, sits with other
fungicides rather than with other morpholines.

The **structural coherence** colour view shows this per molecule: the Tanimoto similarity of its map neighbours
as a share of what its fingerprint neighbours reach. Regions ranked by mean coherence, at the 60-region layer:

| Most structural | Coherence | Least structural | Coherence |
|---|---|---|---|
| Fatty Acyl-CoA Thioesters | 0.92 | Protonated Amine Cations And Chiral Enantiomers | 0.33 |
| Trans-Enoyl And Mixed Acyl-CoA Anions | 0.90 | Substituted Aromatic Xenobiotic Metabolites | 0.34 |
| Hydroxy, Oxo, And Polyunsaturated Fatty Acyl-CoA Anions | 0.89 | Inorganic Metal Salts And Oxoanions | 0.36 |
| Bile Acid CoA Thioesters | 0.81 | Substituted Heterocyclic Aromatic Compounds | 0.36 |
| Sphingolipid And Ceramide N-Acyl Derivatives | 0.79 | Bacterial Amino Acid Metabolites | 0.37 |

Low coherence has two causes: regions defined by function (agrochemicals, kinase inhibitors) and regions of
very small molecules, where the fingerprints themselves carry little information. The hovercard's "nearest by
structure" line lists the two most similar molecules by fingerprint with their Tanimoto similarity, and is
omitted when none reaches 0.3.

### Structural families

The **structural family** colour view runs the comparison the other way. Stage 08 lays the fingerprints out
with their own UMAP (Jaccard metric, otherwise the map's settings), clusters that layout with Toponymy's
clusterer into 112, 29 and 7 families at three granularities, and has Claude name each family from its
members' descriptions and IUPAC names with an instruction to name the shared structure and ignore roles and
sources. The colour view shows the 15 largest of the 29 mid-level families; a third of the molecules belong
to no family dense enough to name and are shown in grey. One name was corrected by hand: the largest family,
5,026 small aromatic and heteroaromatic molecules, came back as "halogenated" although only 28% of its members
carry a halogen.

Coloured onto the description map, a family either stays together or gets sprayed across regions, and that
is the same finding seen per molecule. Counting how many of the map's 60 regions hold 80% of a family:

| Stays together | Regions | Scattered | Regions |
|---|---|---|---|
| Flavonol And Flavone O-Glycosides | 2 | Cyclic Lactams And Pyrrolidinones | 16 |
| Hydroxylated Steroid And Bile Acids | 3 | Substituted Aromatic And Heteroaromatic Compounds | 15 |
| Hydroxy Polyunsaturated Eicosanoid Lipid Mediators | 3 | Anthraquinones And Prenylated Flavanoid Ketones | 14 |
| N-Acetylhexosamine Containing Oligosaccharides | 3 | Indole-3-yl Substituted Compounds | 12 |
| Fatty Acyl-CoA Thioesters | 4 | Aromatic Amino Acid Carboxamides | 11 |

Chemotypes with a systematic naming vocabulary stay together; chemotypes that cut across biological function
are split by it.

## Data and credits

- ChEBI-20 by Carl Edwards, Tuan Lai, Kevin Ros, Garrett Honke, Kyunghyun Cho and Heng Ji, *Translation
  between Molecules and Natural Language* (EMNLP 2022), derived from [ChEBI](https://www.ebi.ac.uk/chebi/)
  (CC BY 4.0) and [PubChem](https://pubchem.ncbi.nlm.nih.gov/).
- Compound names, properties and structure drawings come from PubChem's PUG REST and image services.
- Built with Qwen3-Embedding, UMAP, Toponymy, DataMapPlot, RDKit and Claude.
