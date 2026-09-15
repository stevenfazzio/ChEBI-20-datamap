# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project overview

Two interactive datamaps of ChEBI-20 (Edwards et al. 2022, the MolT5 molecule-captioning benchmark): 33,008
molecules, each a PubChem CID, a SMILES string and a ChEBI natural-language description. The description map
embeds each description, lays the corpus out in 2-d with UMAP, names the regions with Toponymy and renders with
DataMapPlot into `docs/`; the structure map does the same from Morgan fingerprints of the SMILES into
`docs/morgan/`. Each map carries the other's regions as a colour view and a per-molecule score of how far its
neighbourhoods agree with the other similarity. Both publish from `docs/` to GitHub Pages.

## Running the pipeline

```bash
uv sync --extra dev
uv run python pipeline/00_fetch.py          # MolT5's three split files -> data/raw/, data/molecules.parquet
uv run python pipeline/01_enrich.py         # PubChem properties per CID + role fields -> data/corpus.parquet
uv run python pipeline/02_embed.py          # Qwen3-Embedding-4B on the descriptions -> data/embeddings.npz
uv run python pipeline/03_reduce_umap.py    # UMAP -> 2-d, fixed seed -> data/umap_coords.npz
uv run python pipeline/04_label_topics.py   # Toponymy + Claude region names -> data/labels.parquet
uv run python pipeline/07_structure_agreement.py # map neighbourhoods scored by the other similarity -> data/structure_agreement.parquet
uv run python pipeline/08_rhea.py           # Rhea reactions per molecule -> data/rhea.parquet (layout-independent; both maps read it)
uv run python pipeline/06_social_preview.py # static card -> docs/social-preview.png (run before 05)
uv run python pipeline/05_visualize.py      # DataMapPlot -> docs/index.html + docs/chebi20_*.zip
```

Stages 03-07 take `--layout text|morgan`; `morgan` is the structure map (Morgan fingerprints, Jaccard UMAP, `_morgan`
file names, `docs/morgan/`). `make fetch|enrich|embed|umap|label|structure|rhea|preview|visualize|map` wrap the same
commands (`map` runs 02-08 for the description map; `make map-structure` runs 03-07 with `LAYOUT=morgan` and assumes
`data/rhea.parquet` exists); `make lint`,
`make test`, `make serve` (the map fetches its data files relative to its origin, so it must be served,
never opened via `file://`). Useful flags: `01_enrich.py --limit 50` and `02_embed.py --limit 200` are
smoke tests that write nothing; `04_label_topics.py --sweep` reports cluster counts per layer at several granularities with no LLM calls. Stage 04
is the only stage that costs money, once per map. Run it the way `make label` does (`OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1`): on macOS torch,
scikit-learn and numba each load their own `libomp.dylib`, and the multi-runtime OpenMP has deadlocked this
stage in sibling projects.

**Embedding models.** `config.EMBED_MODELS` is a registry keyed by short name; `EMBED_MODEL_KEY` is the
map's model. Stages 02-05 take `--model` / `--embedding <key>`; the default key owns the unsuffixed
artifact names and renders into `docs/`, any other key writes `_<key>`-suffixed copies and renders into
`data/map_<key>/` (`config.keyed_files`).

**Runpod.** `02_embed.py --device runpod` and `04_label_topics.py --device runpod` run the same scripts on a
throwaway GPU pod via `pipeline/remote.py` (runpodctl + ssh) and pull the results back. The Anthropic key
reaches the pod as a remote `.env`, never on a command line. The pod is deleted in a `finally` unless
`--keep-pod`; check `runpodctl pod list` after any crash. Needs `RUNPOD_API_KEY` or `~/.runpod/config.toml`
and the account's registered SSH key at `~/.ssh/id_ed25519`.

Stage scripts are run from the repo root; they import sibling modules (`config`, `io_utils`, `extract`,
`embedder`, `remote`, `structure`, `naming`) because `pipeline/` is `sys.path[0]` when a script there is executed. Tests load stage scripts by path
with importlib, or put `pipeline/` on `sys.path` themselves.

## Data layout

`data/` is gitignored; `docs/` is the published site.

```
data/raw/{train,validation,test}.txt   the MolT5 files as published (tab-separated: CID, SMILES, description)
data/raw/_manifest.json                fetch timestamp, URLs, SHA-256s, row counts
data/molecules.parquet                 cid, smiles, description, split, n_words (stage 00)
data/pubchem_cache/batch_NNNN.json     one PUG REST batch each; the resume unit of stage 01
data/pubchem.parquet                   cid + pubchem_* properties
data/corpus.parquet                    molecules + pubchem + roles/primary_role/primary_organism + name, url, embed_text
data/embeddings.npz                    cid + unit-norm float32 embeddings; embeddings_meta.json records model revision etc.
data/umap_coords.npz                   cid + 2-d layout; umap_meta.json records the parameters
data/umap_coords_morgan.npz            the structure map's layout: 2-d UMAP of the Morgan fingerprints (Jaccard); no model suffix
data/labels.parquet                    cid + cluster_layer_i (int, -1 = unlabelled) + label_layer_i (name); layer 0 is FINEST
data/topic_names.json                  region names per layer; cluster_tree.json the parent edges; labels_meta.json the run record
data/structure_agreement.parquet       cid + coherence, similarity of map/own/other-space neighbours, overlaps, 3 nearest in the other space
data/structure_agreement_meta.json     fingerprint spec, random floor, summary tables, per-region coherence, spread over the other map
data/*_morgan.*                        the structure map's labels, names, tree, meta and agreement files (same schemas)
data/raw/rhea/                         Rhea's public files as fetched (stage 08; _manifest.json has the release and checksums)
data/rhea.parquet                      cid + in_rhea, matched ChEBI ids, n_reactions, is_hub, ec_class, ranked partner cids; rhea_meta.json the run record
docs/morgan/index.html                 the structure map and its own data zips and social-preview.png
docs/index.html                        the map; docs/chebi20_{point,meta,label}_data*.zip its externalised data
```

Every `*.npz` is aligned to `corpus.parquet` row order and carries `cid`; the readers assert it.

Raw files are the provenance record: stage 00 skips any split file that already exists; delete a file to
refetch it. Stage 01 skips any PubChem batch file that already exists.

## Conventions

- Data files are written via `io_utils.write_parquet_safely` / `atomic_write_bytes` (tmp + verify +
  `os.replace`). Never write directly to a live data path.
- Every stage prints its row delta with a reason. An unexplained drop is a bug. Nothing is dropped in v1:
  all 33,008 rows go through every stage.
- `uv` for the environment, `ruff` for lint and format (line length 120, isort with local modules as
  first-party). Python 3.12 (`.python-version`).
- Fixed `random_state` on UMAP; `cvd_safer=True` and glasbey palettes in DataMapPlot; search on a
  composed field; hovercard with name, formula, weight, charge, XLogP, PubChem structure image, the
  description and the two nearest molecules by Morgan fingerprint (stage 07); click opens the PubChem compound page. No histogram, no topic tree. See
  `~/.claude/skills/datamap` for the full defaults and `~/.claude/skills/toponymy` before touching stage 04.

## Decisions so far (2026-09-15)

- **Corpus: all three splits merged**, 26,407 / 3,301 / 3,300 = 33,008 molecules, split kept as a field.
  No filtering: every CID is unique, every description is at least 18 words (the "20" in ChEBI-20 is the
  original word-count floor), and 153 descriptions are shared by two or more CIDs (stereoisomers and
  the like), which is real data rather than duplication.
- **Embedding text = the description alone.** The PubChem name and formula stay out so the layout is driven
  by what ChEBI's prose says (class, structure, role, source), not by identifiers. Every description opens
  "The molecule is"; that shared prefix is left in place.
- **Embedding model: Qwen/Qwen3-Embedding-4B** (Apache 2.0), pinned to a Hub revision, bf16 on GPU,
  unit-normalised, with the instruction "Identify the topic or theme of the given chemical compound
  description" (the form the Qwen3 Embedding report uses for clustering tasks). `qwen3-0.6b` is registered
  for smoke tests and comparison builds. Toponymy's keyphrases go through the same wrapper (`embedder.py`).
- **Clustering happens in the 2-d layout.** One UMAP (n_neighbors 15, min_dist 0.05, cosine, seed 42)
  feeds both the plot and Toponymy's `clusterable_vectors`, so named regions match what a viewer sees.
- **Region names come from Claude Sonnet 5** (chosen over Opus 5 for cost on 2026-09-15; region granularity
  `base_min_cluster_size` 20 chosen the same day) via `LoopSafeNamer` in `04_label_topics.py` (Toponymy's
  `AsyncLiteLLMNamer` with one semaphore per event loop, and `drop_params` because Sonnet 5 rejects the
  temperature parameter Toponymy sends). Detail levels 0.4 -> 0.8 pick the name-length tiers. Names are
  LLM-generated labels and the page says so.
- **Metadata from PubChem, not an LLM.** PUG REST's batch property endpoint gives the display name
  (PubChem's title), IUPAC name, formula, weight, XLogP, charge and H-bond counts for free, and the structure
  image is hot-linked from PubChem's image service (CORS-open) so the hovercard costs nothing in file size.
  An LLM enrichment pass was considered and deferred; nothing in the map needs one yet.
- **Colormaps in v1:** metabolite organism (from "<organism> metabolite" roles; top 12 + Other + None
  stated), first-stated biological role (top 15 + Other + None), formal charge (ordered buckets), log10
  molecular weight, XLogP, dataset split. Roles are stated for 54% of molecules; the "None stated"
  category is pinned to grey rather than dropped.
- **Externalised data.** The map is headed for GitHub Pages and meant to be shared by link, so stage 05
  renders with `inline_data=False`; the zips sit beside `docs/index.html` and are fetched relative to it.
  Open Graph tags are injected into the head; a `docs/social-preview.png`, if present, becomes the card image.
- **Structural agreement, not a second map (2026-09-15).** Stage 07 compares the map with Morgan fingerprints
  (radius 2, 2,048 bits, chirality on, k = 15, exact neighbour search) and feeds stage 05 a "structural
  coherence" colormap (Tanimoto of map neighbours over the fingerprint ceiling, floor 0.05) and a hover line
  naming the two nearest molecules by fingerprint (three are stored; those below Tanimoto 0.3 are omitted;
  the names are searchable because the hover body is the search field). Finding: text and fingerprint
  neighbourhoods agree on 20% of neighbours (median 13%); text neighbours have mean Tanimoto 0.38 against a
  0.58 ceiling and 0.09 floor; agreement tracks molecule size only. Lipid regions are structural, agrochemical
  and small-molecule regions are not. A fingerprint-based map was considered and deferred; the finding says it
  would be a genuinely different view. A structural-family colormap (cluster the fingerprint space, colour
  this map by it) is the natural next step and is already half of that second map.
- **The structure map (2026-09-15).** Stages 03-07 take `--layout`; `config.LAYOUTS` holds what differs (metric,
  granularity, namer text and instruction, overrides). The fingerprint layout is a Jaccard UMAP of the Morgan
  bits with the text map's other settings (helium-3 is disconnected by the approximate neighbour search and parked
  outside the cloud); clustered at `base_min_cluster_size` 20 like the text map (594 / 187 / 50 / 15 regions;
  the sweep gave 783/249/79/22/8 at 15 and 237/74/20/7 at 50) and named by Claude Sonnet 5 from description +
  IUPAC name with an instruction to name shared structure and ignore roles and sources. Raw SMILES were rejected
  as namer input: the keyphrase tokeniser shreds them and the model reads them unreliably. Two overrides
  (`config.STRUCTURE_NAME_OVERRIDES`): regions named "halogenated" at 26% and 36% halogen content. Each map's
  cross colormap and hover line come from the other map's layer with about 30 regions (`CROSS_LAYER_TARGET`:
  the text map's 20, the structure map's 15), so the "structural family" of a text-map hover is a region a
  viewer can find on the structure map. Coherence on both maps is floor-corrected, (map - random) / (ceiling -
  random), because description cosines have a floor of 0.78; that dropped the text map's mean from 0.55 to
  0.44 without reordering regions. An earlier stage 08 that clustered the fingerprint layout coarsely (base 100,
  112/29/7 "families") was superseded by this and its `data/families*` files were deleted. A Murcko-scaffold
  colormap was measured and dropped: 7,619 distinct scaffolds, 25% acyclic, benzene 6.6%, and the next 14
  scaffolds together under 12%, so "Other" would have been most of the map.
- **Rhea overlay, not a third map (2026-09-15).** Property space (RDKit descriptors, PCA 30) and the Rhea reaction
  network were both measured as candidate third axes. Property neighbourhoods overlap fingerprint ones at 0.31
  and text ones at 0.18 with a partner Tanimoto of 0.45: a smoothed structure space organised by size and
  lipophilicity, not a new axis. Rhea covers 40% of the corpus (InChIKey connectivity-layer match, release 142);
  one-hop reaction partners are structural (Tanimoto 0.43-0.48) and only two hops out or at shared-enzyme scale
  does the network diverge from structure (Tanimoto 0.26-0.29). Decision: no third map; Rhea becomes stage 08 and
  an overlay on both maps (enzyme-class colormap with two greys, reaction-count and partner hover line), plus the
  referee number for the README: 29% of partners sit among a molecule's 15 fingerprint neighbours against 13%
  among its description neighbours. Hubs (> 100 reactions) are never partners. The connectivity match also merges
  isotopologues (Glycine-d5 matches glycine) and charge states, by design.

## Data facts (ChEBI-20 as published, fetched 2026-09-15)

- 33,008 rows, all CIDs unique. Description length: median 40 words, p10 25, p90 67, max 166; median 270
  characters, max 1,377.
- Templated clauses: "has a role as" 17,692; "It is a" 26,897; "conjugate base/acid of" 12,117; "derives
  from" 11,310; "isolated from" 3,050. 1,227 distinct roles; 44 distinct metabolite organisms. ChEBI class
  lists are 18,592 distinct strings, so they are search material rather than a colormap.
- SMILES quirk for the future structure map: 4,965 SMILES contain a doubled backslash (`\\`) where the
  published files escaped the bond-direction character. Parse with that in mind; RDKit is not a v1 dependency.
- All 33,008 SMILES parse with RDKit, with or without unescaping the doubled backslash. Morgan r2/2048 with
  chirality gives 29,355 distinct fingerprints (25,836 without: stereoisomers collapse), none empty, median 41
  bits set. 1,522 SMILES are multi-fragment (salts); 237 exceed 500 characters (max 1,598), which matters for
  any SMILES transformer with a token limit.

## Current state (2026-09-15)

The map is built: `docs/index.html` plus `docs/chebi20_*.zip`, 33,008 molecules, 562 / 178 / 60 / 20 / 6 named
regions (base_min_cluster_size 20; the local sweep gave 563, the pod 562, the usual cross-machine wobble). Naming
ran on a Runpod RTX 4090 in 8.0 minutes of Toponymy time with Claude Sonnet 5 (10.9 minutes of pod time); names
have a median of five words at the finest layer and three to four at the coarsest, one duplicate name in layer 1.
Stages 00-03: `data/corpus.parquet` (PubChem returned properties for 32,998 CIDs; XLogP is null for 1,917),
`data/embeddings.npz` (33,008 x 2,560, Qwen3-Embedding-4B in bf16 on a Runpod RTX 5090 at ~145 molecules/s;
the M3 managed 2.3/s), `data/umap_coords.npz` (38 s). Output: `index.html` 0.18 MB plus 6.9 MB of data zips,
of which the hover text is 6.0 MB (the description is stored once; IUPAC names are a third of it). An
`embeddings_qwen3-0.6b.npz` comparison build was started locally the same day and is not part of the map.
Published 2026-09-15: public repo github.com/stevenfazzio/ChEBI-20-datamap, Pages from `docs/`, live at
https://stevenfazzio.com/ChEBI-20-datamap/ (the user site's custom domain), listed on stevenfazzio.com/projects/.

Two things learned on the way: bf16 normalisation leaves norms at 0.998-1.004, so `embedder.py`
renormalises in float32 after the cast (the first pod run failed the unit-norm assertion for that reason);
and Runpod's "no longer any instances available" create error is a placement failure like the others, so
`remote.py` now walks on to the next candidate instead of raising. `04_label_topics.py --preview` writes
placeholder names so stage 05 can be iterated on without LLM calls; stage 05 flags such a build in its subtitle.

Stage 07 added the same day: `data/structure_agreement.parquet` (28 s locally), the structural-coherence colormap and
the nearest-by-structure hover line. Output is now `index.html` 0.18 MB plus 8.0 MB of data zips, of which the hover
text is 7.05 MB (two neighbour names per molecule cost 1.0 MB compressed; a third would cost 0.4 MB more).
The structure map, the same day: `docs/morgan/index.html` plus its own zips (8.58 MB; hover text 7.38 MB) and
social card. Its 846 regions were named in 8.3 min of Toponymy time on a Runpod RTX 4090 (13.1 min of pod time,
about $7 of Sonnet 5 by estimate; not measured). The text map was re-rendered with the structure map's 15
coarsest regions as its "structural family" view (8.45 MB). Stage 04 for the text map was not re-run.

Browser notes (in-app Chromium pane, 800 x 600): the externalised map logs one "deck.gl: assertion failed" (after a
"Pixel project matrix not invertible" warning) on first paint, before the data zips arrive, and then renders and
behaves normally; the inline sibling map does not log it. Treated as benign. "Font IBM Plex Sans did not load" is the
pane blocking Google Fonts. At the overview in that small pane only one or two coarsest labels fit; label density at
a real screen size has not been checked yet.
A freshly opened pane can also leave the canvas blank on its very first load with every data file at 100% and the
layers populated (seen 2026-09-15); navigating away and back paints it. Compare against a reload before
suspecting the build.
