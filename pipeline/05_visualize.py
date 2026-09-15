"""Stage 05: render the interactive map -> docs/index.html + docs/chebi20_*.zip (externalised data).

Reads corpus.parquet, umap_coords.npz, labels.parquet, structure_agreement.parquet (stage 07: the structural
coherence colormap and the nearest-by-structure hover line) and families.parquet (stage 08: the structural-family
colormap and hover line). The data files are written beside the HTML and
fetched relative to it, so the map must be served over HTTP (`make serve`), never opened via file://.
Open Graph tags are added to the page head so a shared link renders as a card.
"""

import argparse
import html
import json
import re

import datamapplot
import glasbey
import numpy as np
import pandas as pd

import config
from io_utils import atomic_write_text

HOVER_TEMPLATE = (
    "<div style=\"font-family:'IBM Plex Sans',sans-serif;width:min(460px,100%);padding:8px 10px;"
    'box-sizing:border-box;color:#1f2328;">'
    f'<img src="{config.PUBCHEM_IMAGE_URL.format(cid="{cid}")}" alt="" loading="lazy" '
    'style="float:right;width:120px;height:120px;margin:0 0 6px 12px;object-fit:contain;background:#fff;'
    'border:1px solid #d0d7de;border-radius:4px;">'
    "{body}"
    '<div style="clear:both;"></div>'
    "</div>"
)
ON_CLICK = f"window.open(`{config.PUBCHEM_COMPOUND_URL.format(cid='{cid}')}`, `_blank`)"

CUSTOM_CSS = """
.deck-tooltip { max-width: min(480px, 92vw) !important; }
/* DataMapPlot sizes the dropdown swatch to its colour count (12px per box, up to five), so a two-category
   colormap gets a 24px swatch and its label sits 36px left of the others. Fix the swatch width and let
   the boxes share it. */
.color-swatch { display: inline-flex !important; width: 60px !important; }
.color-swatch .color-swatch-box { flex: 1 1 0 !important; width: auto !important; }
"""
CUSTOM_JS = """
datamap.deckgl.setProps({controller: {scrollZoom: {speed: 0.05, smooth: true}}});
"""

NEUTRAL = "#bdbdbd"
# Formal charge, read as an ordered scale: blues negative, grey neutral, reds positive.
CHARGE_BUCKETS = [
    ("−3 or lower", "#08306b"),
    ("−2", "#2171b5"),
    ("−1", "#6baed6"),
    ("0", NEUTRAL),
    ("+1", "#fc9272"),
    ("+2", "#ef3b2c"),
    ("+3 or higher", "#a50f15"),
    ("Unknown", "#f0f0f0"),
]


def categorical_palette(n_colors: int) -> list[str]:
    return glasbey.create_palette(
        palette_size=n_colors,
        colorblind_safe=True,
        cvd_severity=50.0,
        lightness_bounds=(25, 75),  # no swatch washes out on a light background
    )


def categorical(
    field: str, description: str, values: np.ndarray, neutral: str | None = None
) -> tuple[dict, np.ndarray]:
    """Colormap metadata for a categorical field; `neutral` names a category pinned to grey."""
    cats = sorted(set(values.tolist()) - {neutral})
    mapping = dict(zip(cats, categorical_palette(len(cats))))
    if neutral is not None and neutral in set(values.tolist()):
        mapping[neutral] = NEUTRAL
    meta = {
        "field": field,
        "description": description,
        "kind": "categorical",
        "color_mapping": mapping,
        "show_legend": True,
    }
    return meta, values


def top_n(values: pd.Series, n: int, other: str, none: str) -> np.ndarray:
    """Keep the n most frequent non-empty values, pool the rest as `other`, and label empties `none`."""
    keep = set(values[values != ""].value_counts().head(n).index)
    return np.where(values == "", none, np.where(values.isin(keep), values, other))


def charge_bucket(charge) -> str:
    if pd.isna(charge):
        return "Unknown"
    c = int(charge)
    if c <= -3:
        return "−3 or lower"
    if c >= 3:
        return "+3 or higher"
    return {-2: "−2", -1: "−1", 0: "0", 1: "+1", 2: "+2"}[c]


def esc(s) -> str:
    return html.escape(str(s) if s is not None and not (isinstance(s, float) and np.isnan(s)) else "")


def formula_html(formula) -> str:
    """C20H31O4- -> C<sub>20</sub>H<sub>31</sub>O<sub>4</sub><sup>−</sup>."""
    if not isinstance(formula, str) or not formula:
        return ""
    m = re.match(r"^(.*?)([+-]\d*)?$", formula)
    body, charge = m.group(1), m.group(2) or ""
    body = re.sub(r"(\d+)", r"<sub>\1</sub>", esc(body))
    if charge:
        charge = charge.replace("-", "−")
        body += f"<sup>{esc(charge)}</sup>"
    return body


def load_label_layers(files: dict, cids: np.ndarray) -> list[np.ndarray]:
    labels = pd.read_parquet(files["labels"])
    assert (labels["cid"].to_numpy() == cids).all(), "labels.parquet is not aligned with corpus.parquet"
    cols = sorted((c for c in labels.columns if c.startswith("label_layer_")), key=lambda c: int(c.rsplit("_", 1)[1]))
    layers = [labels[c].fillna("Unlabelled").to_numpy() for c in cols]
    n_finest, n_coarsest = pd.Series(layers[0]).nunique(), pd.Series(layers[-1]).nunique()
    assert n_finest >= n_coarsest, "label layers must be finest-first for DataMapPlot"
    print(f"{len(layers)} label layers: " + ", ".join(f"{pd.Series(v).nunique() - 1} named" for v in layers))
    return layers


def load_structure(files: dict, cids: np.ndarray) -> pd.DataFrame:
    if not files["structure"].exists():
        raise SystemExit(f"{files['structure']} is missing: run stage 07 (`make structure`) before rendering")
    structure = pd.read_parquet(files["structure"])
    assert (structure["cid"].to_numpy() == cids).all(), "structure_agreement.parquet is not aligned with corpus.parquet"
    return structure


def load_families(files: dict, cids: np.ndarray) -> pd.Series:
    """The structural family of each molecule at the legend layer (the layer with about 30 families)."""
    if not files["families"].exists():
        raise SystemExit(f"{files['families']} is missing: run stage 08 (`make families`) before rendering")
    families = pd.read_parquet(files["families"])
    assert (families["cid"].to_numpy() == cids).all(), "families.parquet is not aligned with corpus.parquet"
    cols = [c for c in families.columns if c.startswith("label_layer_")]
    col = min(cols, key=lambda c: abs(families[c].nunique() - 30))
    print(f"structural families from {col}: {families[col].nunique() - 1} named")
    return families[col]


def nearest_by_structure(structure: pd.DataFrame, names: dict) -> pd.Series:
    """One hover line per molecule naming its nearest fingerprint neighbours, or "" where none is close enough.

    Tiny molecules have near-empty fingerprints and their nearest neighbours are noise, so neighbours below the
    Tanimoto floor are left out and the line disappears exactly where it would mislead.
    """

    def line(cids, sims) -> str:
        shown = zip(cids[: config.STRUCTURE_NEIGHBOURS_SHOWN], sims[: config.STRUCTURE_NEIGHBOURS_SHOWN])
        parts = [f"{esc(names[c])} ({s:.2f})" for c, s in shown if s >= config.STRUCTURE_NEIGHBOUR_MIN_TANIMOTO]
        return "Nearest by structure: " + " · ".join(parts) if parts else ""

    pairs = zip(structure["morgan_neighbour_cids"], structure["morgan_neighbour_tanimoto"])
    return pd.Series([line(c, s) for c, s in pairs])


def build_point_data(
    corpus: pd.DataFrame, nearest: pd.Series | None = None, family: pd.Series | None = None
) -> pd.DataFrame:
    """One HTML column per molecule (`body`) that is both the hovercard and the search text, plus the CID.

    Storing the description once matters: the hover data ships as one gzipped JSON file, and a separate
    search column would double it. Search is a substring match over the column, so the identifiers line at
    the bottom of the card is what makes name, IUPAC name, formula and CID searchable. The nearest-by-structure
    line (`nearest`, from stage 07) and the structural family (`family`, from stage 08) are searchable for the
    same reason: a name also finds the molecules structurally nearest to it, and a family name finds its members.
    """
    corpus = corpus.assign(
        nearest="" if nearest is None else nearest.to_numpy(),
        family="" if family is None else family.replace("Unlabelled", "").to_numpy(),
    )

    def facts_line(row) -> str:
        parts = [formula_html(row["pubchem_molecularFormula"])]
        if pd.notna(row["pubchem_molecularWeight"]):
            parts.append(f"{row['pubchem_molecularWeight']:,.1f} Da")
        if pd.notna(row["pubchem_charge"]):
            c = int(row["pubchem_charge"])
            parts.append("neutral" if c == 0 else f"charge {c:+d}".replace("-", "−"))
        if pd.notna(row["pubchem_xLogP"]):
            parts.append(f"XLogP {row['pubchem_xLogP']:g}")
        return " · ".join(p for p in parts if p)

    def body(row) -> str:
        ids = [esc(row["pubchem_iUPACName"])] if isinstance(row["pubchem_iUPACName"], str) else []
        ids.append(
            " · ".join(
                p
                for p in (
                    esc(row["name"]),
                    esc(row["pubchem_molecularFormula"]),
                    f"PubChem CID {row['cid']}",
                    f"ChEBI-20 {row['split']} split",
                )
                if p
            )
        )
        return (
            '<div style="font-size:14px;font-weight:600;line-height:1.3;overflow-wrap:anywhere;">'
            f"{esc(row['name'])}</div>"
            f'<div style="font-size:11.5px;color:#57606a;margin-top:3px;line-height:1.5;">{facts_line(row)}</div>'
            f'<div style="font-size:12.5px;line-height:1.45;margin-top:8px;">{esc(row["description"])}</div>'
            + "".join(
                f'<div style="font-size:11.5px;color:#57606a;margin-top:6px;line-height:1.5;">{line}</div>'
                for line in (
                    f"Structural family: {esc(row['family'])}" if row["family"] else "",
                    row["nearest"],
                )
                if line
            )
            + '<div style="font-size:11px;color:#8b949e;margin-top:6px;line-height:1.5;overflow-wrap:anywhere;">'
            + "<br>".join(ids)
            + "</div>"
        )

    return pd.DataFrame({"cid": corpus["cid"].to_numpy(), "body": corpus.apply(body, axis=1).to_numpy()})


def add_open_graph(html_path, n_points: int) -> None:
    og = {
        "og:title": config.MAP_TITLE,
        "og:description": (
            f"{n_points:,} molecules from the ChEBI-20 dataset, laid out by the meaning of their ChEBI "
            "descriptions. Pan, zoom, hover and search."
        ),
        "og:type": "website",
        "og:url": config.MAP_URL,
    }
    if (html_path.parent / "social-preview.png").exists():
        og["og:image"] = config.MAP_URL + "social-preview.png"
        og["twitter:card"] = "summary_large_image"
    tags = "\n".join(f'<meta property="{k}" content="{html.escape(v, quote=True)}">' for k, v in og.items())
    text = html_path.read_text(encoding="utf-8")
    assert "<head>" in text, "no <head> in the rendered page"
    atomic_write_text(html_path, text.replace("<head>", "<head>\n" + tags, 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedding", default=config.EMBED_MODEL_KEY, choices=sorted(config.EMBED_MODELS))
    args = ap.parse_args()
    files = config.keyed_files(args.embedding)
    out_dir = files["map_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_parquet(config.PATHS["corpus"])
    cids = corpus["cid"].to_numpy()
    lay = np.load(files["umap"], allow_pickle=True)
    assert (lay["cid"] == cids).all(), "umap_coords.npz is not aligned with corpus.parquet"
    coords = lay["coords"]
    label_layers = load_label_layers(files, cids)
    labels_meta = json.loads(files["labels_meta"].read_text())
    placeholder = labels_meta["namer_model"].startswith("placeholder")
    if placeholder:
        print("labels are placeholders (stage 04 --preview); rendering an unnamed preview")
    structure = load_structure(files, cids)
    family = load_families(files, cids)
    extra = build_point_data(corpus, nearest_by_structure(structure, dict(zip(cids, corpus["name"]))), family)

    organism_meta, organism_vals = categorical(
        "organism",
        "Metabolite of (first organism stated)",
        top_n(corpus["primary_organism"], config.ORGANISM_TOP_N, "Other organism", "None stated"),
        neutral="None stated",
    )
    role_meta, role_vals = categorical(
        "role",
        "Biological role (first stated)",
        top_n(corpus["primary_role"], config.ROLE_TOP_N, "Other role", "None stated"),
        neutral="None stated",
    )
    charge_vals = corpus["pubchem_charge"].map(charge_bucket).to_numpy()
    charge_meta = {
        "field": "charge",
        "description": "Formal charge",
        "kind": "categorical",
        "color_mapping": {k: v for k, v in CHARGE_BUCKETS if k in set(charge_vals.tolist())},
        "show_legend": True,
    }
    # Stage 08: families clustered in a fingerprint layout, named for their shared structure. Coloured on this
    # map, a family that the descriptions scatter by use or source shows up sprayed across several regions.
    family_meta, family_vals = categorical(
        "family",
        f"Structural family (largest {config.FAMILY_LEGEND_TOP_N} of {family.nunique() - 1}, by fingerprint)",
        top_n(family.replace("Unlabelled", ""), config.FAMILY_LEGEND_TOP_N, "Other family", "Unlabelled"),
        neutral="Unlabelled",
    )
    split_meta, split_vals = categorical("split", "ChEBI-20 split", corpus["split"].to_numpy())
    mw = corpus["pubchem_molecularWeight"].astype(float)
    mw_vals = np.log10(mw.fillna(mw.median()).clip(lower=1)).to_numpy()
    mw_meta = {"field": "log_mw", "description": "Molecular weight (log10 Da)", "kind": "continuous", "cmap": "viridis"}
    xlogp = corpus["pubchem_xLogP"].astype(float)
    xlogp_vals = xlogp.fillna(xlogp.median()).clip(-10, 15).to_numpy()
    xlogp_meta = {
        "field": "xlogp",
        "description": f"XLogP (hydrophobicity; {int(xlogp.isna().sum()):,} unknown shown at the median)",
        "kind": "continuous",
        "cmap": "cividis",
    }
    # Stage 07: Tanimoto similarity of a molecule's map neighbours as a share of what its fingerprint neighbours
    # reach. High where the description is effectively a structure (lipids), low where it groups by use or source.
    coherence_vals = structure["coherence"].to_numpy(dtype=float)
    coherence_meta = {
        "field": "coherence",
        "description": (
            "Structural coherence (chemical similarity of map neighbours, 0 to 1; noisy for tiny molecules)"
        ),
        "kind": "continuous",
        "cmap": "plasma",
    }

    plot = datamapplot.create_interactive_plot(
        coords,
        *label_layers,
        hover_text=corpus["name"].tolist(),
        hover_text_html_template=HOVER_TEMPLATE,
        extra_point_data=extra,
        on_click=ON_CLICK,
        enable_search=True,
        search_field="body",
        title=config.MAP_TITLE,
        sub_title=(
            f"{len(corpus):,} molecules from the ChEBI-20 dataset, positioned by the meaning of their ChEBI "
            "descriptions · region names generated by Claude"
            + ("" if args.embedding == config.EMBED_MODEL_KEY else f" · exploration build: {args.embedding}")
            + (" · UNNAMED PREVIEW" if placeholder else "")
        ),
        font_family="IBM Plex Sans",
        cvd_safer=True,
        noise_label="Unlabelled",
        initial_zoom_fraction=config.MAP_INITIAL_ZOOM_FRACTION,
        colormap_rawdata=[
            organism_vals,
            role_vals,
            family_vals,
            charge_vals,
            mw_vals,
            xlogp_vals,
            coherence_vals,
            split_vals,
        ],
        colormap_metadata=[
            organism_meta,
            role_meta,
            family_meta,
            charge_meta,
            mw_meta,
            xlogp_meta,
            coherence_meta,
            split_meta,
        ],
        custom_css=CUSTOM_CSS,
        custom_js=CUSTOM_JS,
        inline_data=False,
        offline_data_path=str(out_dir / config.MAP_DATA_PREFIX),
    )
    html_path = out_dir / "index.html"
    plot.save(str(html_path))
    add_open_graph(html_path, len(corpus))

    print("output sizes:")
    total = 0
    for p in sorted(out_dir.iterdir()):
        if p.is_file() and (p.name == "index.html" or p.name.startswith(config.MAP_DATA_PREFIX + "_")):
            total += p.stat().st_size
            print(f"  {p.stat().st_size / 1e6:8.2f} MB  {p.name}")
    print(f"  {total / 1e6:8.2f} MB  total")
    n_b64 = len(re.findall(r"base64,", html_path.read_text(encoding="utf-8")))
    print(f"base64 blobs still inlined in index.html: {n_b64}")
    print(f"wrote {html_path}; serve with `make serve` and open http://127.0.0.1:8765/")


if __name__ == "__main__":
    main()
