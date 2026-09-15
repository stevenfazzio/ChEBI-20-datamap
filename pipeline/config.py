"""Shared paths, constants, and env loading. Every pipeline script imports from here."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)  # a repo-local .env; keys otherwise arrive through the shell environment (~/.secrets)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DOCS_DIR = ROOT / "docs"

# ── Source: ChEBI-20 (Edwards et al., MolT5, 2022) ───────────────────────────
# The canonical files: three tab-separated splits with CID, SMILES, description, published with MolT5.
CHEBI20_BASE_URL = "https://raw.githubusercontent.com/blender-nlp/MolT5/main/ChEBI-20_data"
CHEBI20_SPLITS = ("train", "validation", "test")
CHEBI20_EXPECTED_ROWS = {"train": 26407, "validation": 3301, "test": 3300}  # the paper's counts; verified 2026-09-15

PATHS = {
    "manifest": RAW_DIR / "_manifest.json",  # fetch timestamp, URLs, checksums, row counts
    "molecules": DATA_DIR / "molecules.parquet",  # stage 00: one row per molecule, all splits
    "pubchem": DATA_DIR / "pubchem.parquet",  # stage 01: PubChem properties per CID
    "pubchem_cache": DATA_DIR / "pubchem_cache",  # stage 01: one JSON per PubChem batch (resume unit)
    "corpus": DATA_DIR / "corpus.parquet",  # stage 01: molecules + PubChem + derived fields + embed_text
    "corpus_meta": DATA_DIR / "corpus_meta.json",
    "rhea": DATA_DIR / "rhea.parquet",  # stage 08: reaction context per molecule; layout-independent
    "rhea_meta": DATA_DIR / "rhea_meta.json",
    "rhea_manifest": RAW_DIR / "rhea" / "_manifest.json",
}

# ── PubChem PUG REST ─────────────────────────────────────────────────────────
PUBCHEM_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_PROPERTIES = [
    "Title",
    "IUPACName",
    "MolecularFormula",
    "MolecularWeight",
    "XLogP",
    "Charge",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "HeavyAtomCount",
]
PUBCHEM_BATCH_SIZE = 200  # CIDs per POST; PUG REST accepts a few hundred comfortably
PUBCHEM_MIN_INTERVAL_S = 0.25  # usage policy: at most 5 requests per second, 400 per minute
PUBCHEM_COMPOUND_URL = "https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
PUBCHEM_IMAGE_URL = "https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid={cid}&t=l"  # 300 px PNG, CORS open

# ── Rhea (stage 08) ──────────────────────────────────────────────────────────
# The expert-curated reaction database whose participants are ChEBI entities: measured biochemistry for a
# ChEBI-derived corpus. Public files, no key. Molecules match Rhea participants by the connectivity layer of the
# InChIKey, since ChEBI-20 lists many compounds in both neutral and charged forms and Rhea uses the pH 7.3 species.
RHEA_BASE_URL = "https://ftp.expasy.org/databases/rhea"
RHEA_FILES = {  # local name -> path under RHEA_BASE_URL
    "rhea-release.properties": "rhea-release.properties",
    "rhea-chebi-smiles.tsv": "tsv/rhea-chebi-smiles.tsv",  # every participant with its SMILES
    "rhea-reactions.txt.gz": "txt/rhea-reactions.txt.gz",  # EQUATION lines list participants by ChEBI id
    "rhea2ec.tsv": "tsv/rhea2ec.tsv",  # reaction -> EC number
}
RHEA_RAW_DIR = RAW_DIR / "rhea"
# A participant in more master reactions than this is a cofactor hub (H+, water, O2, CoA, ATP, NAD ...; 62 entities
# at 100 on the 2026-09 release). Hubs count as "in Rhea" but never as anyone's partner: they would link everything.
RHEA_HUB_MIN_REACTIONS = 100
RHEA_PARTNERS_SHOWN = 3  # partners named in the hovercard; the parquet keeps them all, ranked
EC_CLASSES = {
    "1": "Oxidoreductase",
    "2": "Transferase",
    "3": "Hydrolase",
    "4": "Lyase",
    "5": "Isomerase",
    "6": "Ligase",
    "7": "Translocase",
}

# ── HTTP ─────────────────────────────────────────────────────────────────────
USER_AGENT = "ChEBI-20-datamap/0.1 (research datamap; github.com/stevenfazzio/ChEBI-20-datamap)"
REQUEST_TIMEOUT_S = 60

# ── API keys ─────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Embedding models (stage 02; the same model embeds Toponymy's keyphrases in stage 04) ──
# Qwen3-Embedding's instruction format. The technical report evaluates MTEB clustering tasks with
# "Identify the topic or theme of the given ..." instructions, so this is the intended usage for a
# topic map rather than a retrieval one. The same instruction is applied to every text we embed.
QWEN3_INSTRUCTION = "Instruct: Identify the topic or theme of the given chemical compound description\nQuery: "

# Every model is pinned to a Hugging Face commit. `prompt` is prepended to every text (documents and
# Toponymy's keyphrases alike). `dtype` applies on GPU devices; CPU always runs float32.
EMBED_MODELS = {
    "qwen3-4b": {
        "model": "Qwen/Qwen3-Embedding-4B",  # Apache-2.0; Qwen3 Embedding technical report, 2025
        "revision": "5cf2132abc99cad020ac570b19d031efec650f2b",  # resolved 2026-09-05
        "prompt": QWEN3_INSTRUCTION,
        "max_seq_length": 1024,  # the longest description is 166 words, but IUPAC names tokenize densely
        "batch_size": 16,
        "dtype": "bfloat16",  # 8 GB of weights; fp32 would not leave room on a 24 GB MacBook Air
        "trust_remote_code": False,
        "family": "Qwen3 decoder-only LLM, last-token pooling",
    },
    "qwen3-0.6b": {
        "model": "Qwen/Qwen3-Embedding-0.6B",  # same family, small: pipeline smoke tests and comparison builds
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",  # resolved 2026-09-05
        "prompt": QWEN3_INSTRUCTION,
        "max_seq_length": 1024,
        "batch_size": 32,
        "dtype": "float32",
        "trust_remote_code": False,
        "family": "Qwen3 decoder-only LLM, last-token pooling",
    },
}
EMBED_MODEL_KEY = "qwen3-4b"  # the map's model; other keys write _<key>-suffixed artifacts for comparison
EMBED_CHUNK_SIZE = 1_000  # checkpoint granularity for stage 02


def keyed_files(key: str = EMBED_MODEL_KEY, layout: str = "text") -> dict[str, Path]:
    """Every artifact downstream of stage 01 depends on which embedding model produced it, and everything
    downstream of stage 03 on which layout it belongs to.

    The map's model (EMBED_MODEL_KEY) owns the unsuffixed names; any other key gets `_<key>` suffixed copies, so
    exploration builds coexist with the real map. The description layout ("text") owns the plain names; the
    fingerprint layout ("morgan") gets `_morgan` names and renders into a `morgan/` subdirectory. The fingerprint
    layout itself does not depend on the embedding model, so its coordinates carry no model suffix.
    """
    assert layout in LAYOUTS, layout
    suffix = "" if key == EMBED_MODEL_KEY else f"_{key}"
    lay = "" if layout == "text" else f"_{layout}"
    umap_suffix = suffix if layout == "text" else f"_{layout}"
    map_dir = DOCS_DIR if key == EMBED_MODEL_KEY else DATA_DIR / f"map{suffix}"
    return {
        "npz": DATA_DIR / f"embeddings{suffix}.npz",
        "meta": DATA_DIR / f"embeddings_meta{suffix}.json",
        "cache": DATA_DIR / f"embed_cache{suffix}",
        "umap": DATA_DIR / f"umap_coords{umap_suffix}.npz",
        "umap_meta": DATA_DIR / f"umap_meta{umap_suffix}.json",
        "labels": DATA_DIR / f"labels{lay}{suffix}.parquet",
        "topic_names": DATA_DIR / f"topic_names{lay}{suffix}.json",
        "cluster_tree": DATA_DIR / f"cluster_tree{lay}{suffix}.json",
        "labels_meta": DATA_DIR / f"labels_meta{lay}{suffix}.json",
        "structure": DATA_DIR / f"structure_agreement{lay}{suffix}.parquet",
        "structure_meta": DATA_DIR / f"structure_agreement_meta{lay}{suffix}.json",
        # Exploration builds render into data/; the map's model renders straight into docs/ (stage 05).
        "map_dir": map_dir if layout == "text" else map_dir / layout,
    }


def other_layout(layout: str) -> str:
    return "morgan" if layout == "text" else "text"


# ── Runpod (stages 02 and 04 with --device runpod) ───────────────────────────
RUNPOD_POD_NAME = "chebi20-embed"
RUNPOD_TEMPLATE_ID = "runpod-torch-v280"  # runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404, torch in system python
RUNPOD_GPU_ID = "NVIDIA GeForce RTX 4090"  # 24 GB: the 4B model in bf16 is ~8 GB
# Tried in order, each in the data center that currently reports the most stock. A pod created without a
# data-center constraint can be "rented" in a site with no machine and never get scheduled (seen 2026-09-05).
RUNPOD_GPU_CANDIDATES = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 5090",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA RTX A6000",
    "NVIDIA L40S",
    "NVIDIA A40",
    "NVIDIA L4",
]
RUNPOD_CLOUD_TYPE = "SECURE"
RUNPOD_CONTAINER_DISK_GB = 40
RUNPOD_REMOTE_DIR = "/workspace/ChEBI-20-datamap"
RUNPOD_HF_HOME = "/workspace/hf-cache"
# Installed into the template's system python, which already has torch + CUDA. Pinned to the local versions
# (see uv.lock) so the pod embeds with exactly what stage 04 later loads for keyphrases.
RUNPOD_PIP_PACKAGES = [
    "sentence-transformers==5.7.0",
    "transformers==4.57.6",
    "pandas>=2.2",
    "pyarrow>=16",
    "python-dotenv>=1.0",
]
# Stage 04 on a pod additionally needs Toponymy and its LLM client, pinned to the local versions.
RUNPOD_LABEL_PACKAGES = ["toponymy==0.5.4", "litellm==1.101.0"]

# ── Layout (stage 03) ────────────────────────────────────────────────────────
UMAP_N_NEIGHBORS = 15
# min_dist is a clustering parameter here, not a cosmetic one: Toponymy clusters in this 2-d space,
# so regions need to come out dense enough for density clustering to find them. Keep it low.
UMAP_MIN_DIST = 0.05
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42  # fixed seed: forces UMAP to a single thread, worth it for a stable layout

# ── Region naming (stage 04) ─────────────────────────────────────────────────
TOPONYMY_MIN_CLUSTERS = 6
TOPONYMY_BASE_MIN_CLUSTER_SIZE = 20  # 33k points; `04_label_topics.py --sweep` reports the alternatives
# Detail levels pick the name-length tier of Toponymy's prompts per layer: each layer's level is spread over
# [lowest, highest] and rounded onto one of seven tiers (toponymy.templates.SUMMARY_KINDS). The library defaults
# (0.0 -> 1.0) ask the finest layer for 8-to-15-word names, which DataMapPlot wraps to many lines. With four
# layers, 0.4 -> 0.8 gives "4 to 8", "3 to 6", "2 to 5" and "1 to 4" words, finest to coarsest.
TOPONYMY_LOWEST_DETAIL = 0.4
TOPONYMY_HIGHEST_DETAIL = 0.8
NAMER_MODEL = "claude-sonnet-5"  # chosen 2026-09-15 over Opus 5 for cost; the sibling course map used Opus 5
NAMER_CONCURRENCY = 8
# Sonnet 5 (like Opus 5) rejects the temperature parameter Toponymy always sends; litellm's drop_params removes
# it, so naming runs at the model default (temperature 1.0). Names are stored, not regenerated, per run.
NAMER_PROVIDER_KWARGS = {"drop_params": True}
# Appended to Toponymy's naming prompts. Style only; it does not change which molecules form a region.
NAMER_STYLE = (
    "Write topic names in title case, with no colon, subtitle, or list of examples. Name the chemical class, "
    "biological role, or source organism that unites the group rather than listing individual compounds."
)
OBJECT_DESCRIPTION = "natural-language descriptions of chemical compounds from the ChEBI database"
CORPUS_DESCRIPTION = (
    "the ChEBI-20 dataset: 33,008 molecules paired with their ChEBI descriptions, which state each "
    "compound's chemical class, structural relationships, biological roles and source organisms"
)

# ── Agreement between the two layouts (stage 07) ─────────────────────────────
# Morgan fingerprints of the SMILES, the cheminformatics baseline for "structurally similar". Radius 2 with
# 2,048 bits is the ECFP4-equivalent everyone compares against. Chirality is on because 20,039 SMILES carry
# stereocentres and stereoisomer pairs are common here; without it they share one fingerprint.
MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048
MORGAN_CHIRALITY = True
AGREEMENT_K = UMAP_N_NEIGHBORS  # neighbourhood size for every space compared; matches the layouts' own
# coherence = how far a molecule's map neighbours are, in the complementary space (fingerprints for the text
# map, descriptions for the fingerprint map), from chance towards the best that space offers:
# (similarity of map neighbours - random floor) / (similarity of that space's own nearest neighbours - floor).
# Tiny ions have near-empty fingerprints and a ceiling at the floor; the minimum span keeps the ratio finite.
COHERENCE_MIN_SPAN = 0.05
STRUCTURE_NEIGHBOURS_STORED = 3  # nearest complementary-space neighbours kept per molecule (cheap in the parquet)
# Shown in the hovercard. Each name costs ~0.45 MB of compressed hover data (measured 2026-09-15: three names
# added 1.35 MB to a 6 MB file, two added 0.95 MB); the similarity values beside them cost almost nothing.
STRUCTURE_NEIGHBOURS_SHOWN = 2

# ── Layouts ──────────────────────────────────────────────────────────────────
# Two maps of the same corpus: "text" lays the molecules out by their description embeddings, "morgan" by Morgan
# fingerprints of their SMILES. Stages 03-07 take `--layout`; the settings that differ live here. Region names on
# both maps come from the descriptions (the namer reads text), but the fingerprint map's namer also sees the IUPAC
# name and is told to name shared structure, since its regions are structural clusters.
STRUCTURE_NAMER_STYLE = (
    "Write names in title case, with no colon, subtitle, or list of examples. These groups were formed by "
    "similarity of the molecules' Morgan fingerprints, so name the chemical structure the members share (their "
    "scaffold, ring system, functional groups or structural class), never a biological role, use or source organism."
)
STRUCTURE_OBJECT_DESCRIPTION = (
    "chemical compounds, each given by its ChEBI description and IUPAC name, grouped by structural similarity"
)
# Hand corrections to Claude's names, keyed by (layer, the name it produced) -> (name used, reason). Applied when
# stage 04 writes its outputs and recorded in the labels meta; the map says names are generated.
STRUCTURE_NAME_OVERRIDES: dict[tuple[int, str], tuple[str, str]] = {
    (3, "Halogenated Aromatic Compounds"): (
        "Substituted Aromatic And Heteroaromatic Compounds",
        "2,497 molecules, the diffuse mass of small aromatic drugs, pesticides, dyes and benzenoids (median 16 heavy "
        "atoms); 26% carry a halogen, three times the corpus rate but a minority (2026-09-15)",
    ),
    (2, "Halogenated Benzoic Acids And Amides"): (
        "Substituted Benzoic Acids And Benzamides",
        "538 molecules; 36% carry a halogen (2026-09-15)",
    ),
}
LAYOUTS = {
    "text": {
        "metric": UMAP_METRIC,
        "min_clusters": TOPONYMY_MIN_CLUSTERS,
        "base_min_cluster_size": TOPONYMY_BASE_MIN_CLUSTER_SIZE,
        "detail_levels": (TOPONYMY_LOWEST_DETAIL, TOPONYMY_HIGHEST_DETAIL),
        "namer_style": NAMER_STYLE,
        "object_description": OBJECT_DESCRIPTION,
        "name_overrides": {},
        "namer_sees_iupac": False,
        "sweep": [(6, 15), (6, 20), (6, 30), (6, 50), (4, 20)],
        "positioned_by": "the meaning of their ChEBI descriptions",
        "complement": "structure",  # what stage 07 scores the map's neighbourhoods by
        "neighbour_min_similarity": 0.3,  # below this Tanimoto a "nearest" molecule is noise (1.7% have none above)
    },
    "morgan": {
        "metric": "jaccard",
        "min_clusters": TOPONYMY_MIN_CLUSTERS,
        "base_min_cluster_size": 20,  # `04_label_topics.py --layout morgan --sweep` reports the alternatives
        "detail_levels": (TOPONYMY_LOWEST_DETAIL, TOPONYMY_HIGHEST_DETAIL),
        "namer_style": STRUCTURE_NAMER_STYLE,
        "object_description": STRUCTURE_OBJECT_DESCRIPTION,
        "name_overrides": STRUCTURE_NAME_OVERRIDES,
        "namer_sees_iupac": True,
        "sweep": [(6, 15), (6, 20), (6, 30), (6, 50), (6, 100)],
        "positioned_by": "the similarity of their chemical structures (Morgan fingerprints)",
        "complement": "description",
        "neighbour_min_similarity": 0.0,  # cosine between descriptions; every molecule has a meaningful nearest
    },
}
CROSS_LEGEND_TOP_N = 15  # regions of the other map shown by name in a colormap; the rest pool as "Other"
CROSS_LAYER_TARGET = 30  # the other map's layer with about this many regions supplies that colormap

# ── Rendering (stage 05) ─────────────────────────────────────────────────────
MAP_TITLE = "ChEBI-20 Molecule Map"
MAP_DATA_PREFIX = "chebi20"  # docs/chebi20_point_data_0.zip and friends sit beside index.html
MAP_URL = "https://stevenfazzio.com/ChEBI-20-datamap/"  # the user site carries the domain; github.io redirects here


def map_url(layout: str) -> str:
    return MAP_URL if layout == "text" else MAP_URL + layout + "/"


def map_title(layout: str) -> str:
    return MAP_TITLE if layout == "text" else f"{MAP_TITLE}, by structure"


MAP_INITIAL_ZOOM_FRACTION = 0.98
ORGANISM_TOP_N = 12  # categories kept in the metabolite-organism colormap; the rest pool as "Other organism"
ROLE_TOP_N = 15
