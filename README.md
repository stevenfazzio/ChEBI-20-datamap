# ChEBI-20 Molecule Map

Two interactive maps of the [ChEBI-20](https://github.com/blender-nlp/MolT5) dataset: 33,008 molecules, each
placed by the meaning of its ChEBI description on one map and by its chemical structure on the other, with
regions named at four zoom levels and each map showing where the other disagrees with it.

**Live maps:** https://stevenfazzio.com/ChEBI-20-datamap/ (by description) and
https://stevenfazzio.com/ChEBI-20-datamap/morgan/ (by structure)

## What the maps show

ChEBI-20 (Edwards et al. 2022) pairs a PubChem compound ID and SMILES string with the natural-language
description ChEBI curators wrote for the molecule: its chemical class, how it relates to other compounds,
its biological roles, and where it was isolated. This map embeds those descriptions and lays them out in
two dimensions, so molecules that ChEBI describes similarly sit near each other. Hover for the compound
name, formula, weight, charge, XLogP, structure drawing and full description, then four labelled rows in the
same place on every card: the molecule's region (the finest named one it belongs to), its structural family, the
two molecules nearest to it by chemical structure, and its Rhea reactions and partners. Click to open the
PubChem page; search by name, IUPAC name, formula, CID, region, family, neighbour or any phrase in the
description. The colour menu switches between the region colouring and nine metadata views:

| Colour view | What it shows |
|---|---|
| Metabolite organism (first stated) | The organism of the first "<organism> metabolite" role in the description, top 12 named and the rest pooled; 26% of molecules state one. |
| Biological role (first stated) | The first role in the "has a role as" clause, top 15 named and the rest pooled; 54% of molecules state one. "First" is ChEBI's order, which leans general-first (human before mouse, metabolite before anything) and is not a ranking of importance. |
| Enzyme class (Rhea) | The commonest EC class among the molecule's Rhea reactions; two greys for outside Rhea and in Rhea without an EC number. |
| Structural family | The structure map's 15 coarsest regions; a third of molecules belong to no region dense enough to name and are grey. |
| Formal charge | PubChem's formal charge, bucketed from −3 or lower to +3 or higher. |
| Molecular weight (log10 Da) | The colorbar is in log10 units: 2 is 100 Da, 3 is 1,000 Da. |
| XLogP | PubChem's computed hydrophobicity; the 1,917 molecules without one are drawn at the median. |
| Structural coherence | How structurally alike a molecule's map neighbours are, 0 to 1 (explained below); noisy for tiny molecules. |
| Dataset split | ChEBI-20's train, validation and test split. |

The second map lays the same molecules out by Morgan fingerprints of their SMILES, so molecules with similar
substructures sit near each other regardless of what ChEBI says about them, and its regions are named for the
structure their members share. Its views mirror the first map's: the hovercard names the two molecules nearest
by description (without scores: description cosines floor at 0.78, so every score would read 0.9) and the
description-map region, and the colour menu offers "description-map region" and "description coherence". Each
map links to the other and to this repository from its subtitle. The cross views and the coherence scores are
explained below.

## How it is built

| Stage | Script | What it does |
|---|---|---|
| 00 | `pipeline/00_fetch.py` | Downloads the three MolT5 split files and merges them (train 26,407 / validation 3,301 / test 3,300). |
| 01 | `pipeline/01_enrich.py` | Fetches PubChem properties for every CID in batches; reads role and organism fields off the description prose. |
| 02 | `pipeline/02_embed.py` | Embeds each description with [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B), locally or on a Runpod GPU. |
| 03 | `pipeline/03_reduce_umap.py` | UMAP to two dimensions with a fixed seed: of the embeddings, or (`--layout morgan`) of Morgan fingerprints of the SMILES ([RDKit](https://www.rdkit.org/)) under the Jaccard metric. |
| 04 | `pipeline/04_label_topics.py` | [Toponymy](https://github.com/TutteInstitute/toponymy) clusters a layout and names the regions with Claude; for the structure map the namer also reads IUPAC names and is told to name shared structure. |
| 07 | `pipeline/07_structure_agreement.py` | Scores each molecule's map neighbours by the similarity the map does not show (fingerprints for the description map, descriptions for the structure map) and finds its nearest neighbours in that space. |
| 08 | `pipeline/08_rhea.py` | Fetches [Rhea](https://www.rhea-db.org/) and matches every molecule to its reactions: reaction count, cofactor hubs, dominant enzyme class, reaction partners inside the corpus. |
| 05 | `pipeline/05_visualize.py` | [DataMapPlot](https://github.com/TutteInstitute/datamapplot) renders a map into `docs/` or `docs/morgan/`, with the other map's regions as a colour view. |

```bash
uv sync --extra dev
make fetch enrich embed umap label structure rhea visualize   # or: make map (stages 02-08, the description map)
make map-structure                                      # the same stages on the fingerprint layout -> docs/morgan/
make serve                                              # http://127.0.0.1:8765/ and /morgan/
```

Stage 04 calls the Anthropic API and is the only stage that costs money, once per map. `make label` runs it
with the environment variables it needs on macOS. Region names are LLM-generated labels for clusters; they
summarise what a region's molecules have in common and are not ChEBI classifications.

## How structural is the description map?

Its layout comes from the descriptions alone, so it is fair to ask how far it agrees with chemical structure.
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

The **structural coherence** colour view shows this per molecule: how far the Tanimoto similarity of its map
neighbours sits between a random neighbourhood (0) and its own fingerprint neighbours (1). Regions ranked by
mean coherence, at the 60-region layer:

| Most structural | Coherence | Least structural | Coherence |
|---|---|---|---|
| Fatty Acyl-CoA Thioesters | 0.91 | Substituted Heterocyclic Aromatic Compounds | 0.15 |
| Trans-Enoyl And Mixed Acyl-CoA Anions | 0.88 | Protonated Amine Cations And Chiral Enantiomers | 0.15 |
| Hydroxy, Oxo, And Polyunsaturated Fatty Acyl-CoA Anions | 0.87 | Substituted Aromatic Xenobiotic Metabolites | 0.16 |
| Bile Acid CoA Thioesters | 0.78 | Herbicide And Fungicide Agrochemicals | 0.19 |
| Sphingolipid And Ceramide N-Acyl Derivatives | 0.77 | Inorganic Metal Salts And Oxoanions | 0.20 |

Low coherence has two causes: regions defined by function (agrochemicals, kinase inhibitors) and regions of
very small molecules, where the fingerprints themselves carry little information. The hovercard's "nearest by
structure" line lists the two most similar molecules by fingerprint with their Tanimoto similarity, and is
omitted when none reaches 0.3.

## The structure map

The second map answers the same question from the other side. Stage 03 lays the Morgan fingerprints out with
their own UMAP (Jaccard metric, otherwise the first map's settings), stage 04 clusters that layout into 594,
187, 50 and 15 regions, and Claude names each region from its members' descriptions and IUPAC names with an
instruction to name the shared structure and ignore roles and sources. Two names were corrected by hand: the
largest region, 2,497 small aromatic and heteroaromatic molecules, and a benzoic-acid region both came back as
"halogenated" although only 26% and 36% of their members carry a halogen.

Stage 07 then scores the structure map by descriptions: the **description coherence** view is how far the
cosine similarity of a molecule's map neighbours' description embeddings sits between a random neighbourhood
(0) and its own nearest descriptions (1). The random floor matters here, since any two ChEBI descriptions
already have a cosine similarity near 0.78. The 15 coarsest regions, with the number of the description map's
60 regions it takes to hold 80% of each:

| Structure-map region | Molecules | Description coherence | Description-map regions for 80% |
|---|---|---|---|
| Fatty Acyl-CoA Thioesters | 1,208 | 0.91 | 4 |
| Acetamido Amino Sugar Oligosaccharides | 986 | 0.83 | 3 |
| Flavonoid O- And C-Glycosides | 819 | 0.78 | 4 |
| Hydroperoxy And Hydroxy Icosanoid Fatty Acids | 1,546 | 0.76 | 7 |
| Steroid And Triterpenoid Sterols | 1,105 | 0.73 | 4 |
| Glycosidically Linked Pyranose Oligosaccharides | 1,131 | 0.70 | 7 |
| Alpha-Amino Acid Zwitterions | 580 | 0.69 | 4 |
| Nucleoside Phosphate Derivatives | 1,217 | 0.69 | 5 |
| Prenylated Flavonoids And Xanthones | 1,724 | 0.65 | 9 |
| Hydroxy And Oxo Carboxylic Acids | 506 | 0.63 | 8 |
| Sesquiterpenoid Lactones And Terpenoids | 721 | 0.62 | 6 |
| Inorganic And Organic Oxoanions | 2,031 | 0.46 | 11 |
| Heterocyclic Sulfonamide Scaffolds | 716 | 0.45 | 5 |
| Methoxyphenyl Cinnamic Acid Derivatives | 529 | 0.38 | 12 |
| Substituted Aromatic And Heteroaromatic Compounds | 2,497 | 0.30 | 14 |

The mirror holds too. The description map's regions defined by function are the ones the structure map
scatters: "Protonated Alkaloid Ammonium Cations" spreads over 14 structure-map regions, "CNS And Cardiovascular
Drugs" over 13 and "Kinase Inhibitor Antineoplastic Agents" over 11, while "Nucleotide And Nucleoside Phosphate
Derivatives" needs 2 and "Hydroxy And Oxo Fatty Acids" 3. Chemotypes with a systematic naming vocabulary stay
together on both maps; chemotypes that cut across biological function are split by it.

Each map carries the other's regions as a colour view and a hovercard line: the description map shows the
structure map's 15 coarsest regions as **structural family** (a third of molecules belong to no region dense
enough to name and are shown in grey), and the structure map shows the description map's 20 coarsest regions
as **description-map region**.

## Measured biochemistry: what Rhea adds

Both maps are built from what a molecule is or how it is described. [Rhea](https://www.rhea-db.org/), the
expert-curated reaction database whose participants are ChEBI entities, adds what a molecule does. Stage 08 matches
ChEBI-20 to Rhea by the connectivity layer of the InChIKey (ChEBI-20 lists many compounds in both neutral and
charged forms) and, over Rhea's 18,611 master reactions, records each molecule's reaction count, whether it is a
cofactor hub (a participant in more than 100 reactions: water, ATP, NAD, CoA and 58 others), the commonest enzyme
class among its reactions, and its reaction partners inside the corpus, other participants in the same reactions
with hubs and other charge states of the same compound excluded.

| | Molecules |
|---|---|
| In at least one Rhea reaction | 13,314 (40%) |
| With a reaction partner inside ChEBI-20 | 12,402 (38%) |
| Cofactor hubs | 192 |

The **enzyme class** colour view shows the dominant class on both maps, with two greys: outside Rhea, and in Rhea
but in reactions that carry no EC number. The hovercard line gives the reaction count and up to three partners,
ranked by reactions shared.

Reaction partners also settle a question neither map could answer alone, since roles are in the descriptions and
properties follow from structure. A reaction partner is one enzymatic step away, and the fingerprint map predicts
that step far better than the description map does:

| Where a molecule's reaction partners turn up | Share of partners |
|---|---|
| Among its 15 nearest fingerprint neighbours | 29% |
| Among its 15 nearest description neighbours | 13% |

Partners have a mean Tanimoto similarity of 0.43 to each other (random pairs 0.10) and a description cosine of 0.91
(random 0.80). Two hops along the network the structural similarity halves, so the reaction context carries
information the fingerprints do not, but only at pathway scale and only for the two fifths of the corpus that Rhea
covers, which is why it is an overlay on both maps rather than a third map.

## Data and credits

- ChEBI-20 by Carl Edwards, Tuan Lai, Kevin Ros, Garrett Honke, Kyunghyun Cho and Heng Ji, *Translation
  between Molecules and Natural Language* (EMNLP 2022), derived from [ChEBI](https://www.ebi.ac.uk/chebi/)
  (CC BY 4.0) and [PubChem](https://pubchem.ncbi.nlm.nih.gov/).
- Compound names, properties and structure drawings come from PubChem's PUG REST and image services.
- Reaction context comes from [Rhea](https://www.rhea-db.org/) release 142 (CC BY 4.0).
- Built with Qwen3-Embedding, UMAP, Toponymy, DataMapPlot, RDKit and Claude.
