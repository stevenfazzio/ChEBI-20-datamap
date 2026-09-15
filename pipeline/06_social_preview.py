"""Stage 06: a static render for sharing -> docs/social-preview.png (Open Graph card, 1200 x 630).

Reads the same layout and labels as stage 05 and draws only the coarsest layer, which is what reads at card size.
Stage 05 picks the PNG up as the og:image when it exists, so run this before the final `05_visualize.py`.
The project thumbnail on stevenfazzio.com is captured from the live map by that site's own script instead.
"""

import argparse

import datamapplot
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import config

OG_SIZE = (1200, 630)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedding", default=config.EMBED_MODEL_KEY, choices=sorted(config.EMBED_MODELS))
    args = ap.parse_args()
    files = config.keyed_files(args.embedding)

    coords = np.load(files["umap"], allow_pickle=True)["coords"]
    labels = pd.read_parquet(files["labels"])
    cols = sorted((c for c in labels.columns if c.startswith("label_layer_")), key=lambda c: int(c.rsplit("_", 1)[1]))
    coarsest = labels[cols[-1]].fillna("Unlabelled").to_numpy()
    print(f"drawing {len(coords):,} points with the {pd.Series(coarsest).nunique() - 1} coarsest labels")

    fig, _ = datamapplot.create_plot(
        coords,
        coarsest,
        title=config.MAP_TITLE,
        sub_title=f"{len(coords):,} molecules, positioned by the meaning of their ChEBI descriptions",
        noise_label="Unlabelled",
        cvd_safer=True,
        figsize=(OG_SIZE[0] / 100, OG_SIZE[1] / 100),
        dpi=100,
        label_font_size=11,
        label_wrap_width=18,
        title_keywords={"fontsize": 26},
        sub_title_keywords={"fontsize": 12},
        point_size=2.5,
        use_medoids=True,
    )
    png = files["map_dir"] / "social-preview.png"
    # The title sits outside the axes, so a plain save at the figure size clips it; save tight, then pad back
    # to the card size on white.
    fig.savefig(png, dpi=100, bbox_inches="tight", facecolor="white")
    with Image.open(png) as im:
        card = ImageOps.pad(im.convert("RGB"), OG_SIZE, color="white")
    card.save(png, optimize=True)
    print(f"wrote {png} {card.size} ({png.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
