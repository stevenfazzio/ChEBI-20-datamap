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


def keyed_files(key: str = EMBED_MODEL_KEY) -> dict[str, Path]:
    """Every artifact downstream of stage 01 depends on which embedding model produced it.

    The map's model (EMBED_MODEL_KEY) owns the unsuffixed names; any other key gets `_<key>` suffixed
    copies, so exploration builds coexist with the real map.
    """
    suffix = "" if key == EMBED_MODEL_KEY else f"_{key}"
    return {
        "npz": DATA_DIR / f"embeddings{suffix}.npz",
        "meta": DATA_DIR / f"embeddings_meta{suffix}.json",
        "cache": DATA_DIR / f"embed_cache{suffix}",
        "umap": DATA_DIR / f"umap_coords{suffix}.npz",
        "umap_meta": DATA_DIR / f"umap_meta{suffix}.json",
        "labels": DATA_DIR / f"labels{suffix}.parquet",
        "topic_names": DATA_DIR / f"topic_names{suffix}.json",
        "cluster_tree": DATA_DIR / f"cluster_tree{suffix}.json",
        "labels_meta": DATA_DIR / f"labels_meta{suffix}.json",
        # Exploration builds render into data/; the map's model renders straight into docs/ (stage 05).
        "map_dir": DOCS_DIR if key == EMBED_MODEL_KEY else DATA_DIR / f"map{suffix}",
    }


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

# ── Rendering (stage 05) ─────────────────────────────────────────────────────
MAP_TITLE = "ChEBI-20 Molecule Map"
MAP_DATA_PREFIX = "chebi20"  # docs/chebi20_point_data_0.zip and friends sit beside index.html
MAP_URL = "https://stevenfazzio.com/ChEBI-20-datamap/"  # the user site carries the domain; github.io redirects here
MAP_INITIAL_ZOOM_FRACTION = 0.98
ORGANISM_TOP_N = 12  # categories kept in the metabolite-organism colormap; the rest pool as "Other organism"
ROLE_TOP_N = 15
