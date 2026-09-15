# ChEBI-20 Molecule Map

An interactive map of the [ChEBI-20](https://github.com/blender-nlp/MolT5) dataset: 33,008 molecules, each
placed by the meaning of its ChEBI description, with regions named at four zoom levels.

**Live map:** https://stevenfazzio.com/ChEBI-20-datamap/

## What the map shows

ChEBI-20 (Edwards et al. 2022) pairs a PubChem compound ID and SMILES string with the natural-language
description ChEBI curators wrote for the molecule: its chemical class, how it relates to other compounds,
its biological roles, and where it was isolated. This map embeds those descriptions and lays them out in
two dimensions, so molecules that ChEBI describes similarly sit near each other. Hover for the compound
name, formula, weight, charge, XLogP, structure drawing and full description; click to open the PubChem
page; search by name, formula, CID or any phrase in the description. The colour menu switches between
the region colouring and six metadata views: metabolite organism, first-stated biological role, formal
charge, molecular weight, XLogP and dataset split.

## How it is built

| Stage | Script | What it does |
|---|---|---|
| 00 | `pipeline/00_fetch.py` | Downloads the three MolT5 split files and merges them (train 26,407 / validation 3,301 / test 3,300). |
| 01 | `pipeline/01_enrich.py` | Fetches PubChem properties for every CID in batches; reads role and organism fields off the description prose. |
| 02 | `pipeline/02_embed.py` | Embeds each description with [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B), locally or on a Runpod GPU. |
| 03 | `pipeline/03_reduce_umap.py` | UMAP to two dimensions with a fixed seed. |
| 04 | `pipeline/04_label_topics.py` | [Toponymy](https://github.com/TutteInstitute/toponymy) clusters the 2-d layout and names the regions with Claude. |
| 05 | `pipeline/05_visualize.py` | [DataMapPlot](https://github.com/TutteInstitute/datamapplot) renders the map into `docs/`. |

```bash
uv sync --extra dev
make fetch enrich embed umap label visualize   # or: make map (stages 02-05)
make serve                                     # http://127.0.0.1:8765/
```

Stage 04 calls the Anthropic API and is the only stage that costs money. `make label` runs it with the
environment variables it needs on macOS. Region names are LLM-generated labels for clusters of
descriptions; they summarise what a region's molecules have in common and are not ChEBI classifications.

## Data and credits

- ChEBI-20 by Carl Edwards, Tuan Lai, Kevin Ros, Garrett Honke, Kyunghyun Cho and Heng Ji, *Translation
  between Molecules and Natural Language* (EMNLP 2022), derived from [ChEBI](https://www.ebi.ac.uk/chebi/)
  (CC BY 4.0) and [PubChem](https://pubchem.ncbi.nlm.nih.gov/).
- Compound names, properties and structure drawings come from PubChem's PUG REST and image services.
- Built with Qwen3-Embedding, UMAP, Toponymy, DataMapPlot and Claude.
