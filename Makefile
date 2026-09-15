.PHONY: install lint format test fetch enrich embed umap label visualize preview map serve clean

install:
	uv sync --extra dev

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff format .

test:
	uv run pytest

fetch:
	uv run python pipeline/00_fetch.py

enrich:
	uv run python pipeline/01_enrich.py

embed:
	uv run python pipeline/02_embed.py

umap:
	uv run python pipeline/03_reduce_umap.py

# HF_HUB_OFFLINE: the model is cached by stage 02, and a live Hub check has hung this stage before.
# OMP_NUM_THREADS=1: torch, scikit-learn and numba each load their own libomp on macOS, and the
# resulting multi-runtime OpenMP has deadlocked this stage in sibling projects; the heavy work is on the GPU anyway.
label:
	OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
		uv run python pipeline/04_label_topics.py

# Run before visualize: stage 05 uses the PNG as the Open Graph image when it exists.
preview:
	uv run python pipeline/06_social_preview.py

visualize:
	uv run python pipeline/05_visualize.py

map: embed umap label preview visualize

# The map fetches its data files relative to its origin, so it must be served, never opened via file://.
serve:
	cd docs && python3 -m http.server 8765 --bind 127.0.0.1

clean:
	@echo "This will remove all files in data/. Press Ctrl+C to cancel."
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf data/*
