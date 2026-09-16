"""Stage 05: render an interactive map -> docs/index.html + docs/chebi20_*.zip (externalised data), or docs/morgan/
for the structure map (`--layout morgan`).

Reads the corpus, the layout, its labels and its stage 07 agreement file (the coherence colormap and the
nearest-in-the-other-space hover line), the other map's labels for a cross colormap and hover line (the text map
shows each molecule's structural family, the structure map its description-map region), and stage 08's Rhea
context (an enzyme-class colormap and a reaction-partners hover line, the same on both maps). The data files are
written beside the HTML and fetched relative to it, so the map must be served over HTTP (`make serve`), never
opened via file://. Open Graph tags are added to the page head so a shared link renders as a card, and the
subtitle links to the other map.
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
    '<div class="hc">'
    f'<img class="hc-img" src="{config.PUBCHEM_IMAGE_URL.format(cid="{cid}")}" alt="" loading="lazy">'
    "{body}"
    '<div style="clear:both;"></div>'
    "</div>"
)
ON_CLICK = f"window.open(`{config.PUBCHEM_COMPOUND_URL.format(cid='{cid}')}`, `_blank`)"

CUSTOM_CSS = """
.deck-tooltip { max-width: min(480px, 92vw) !important; }
/* The hovercard. Classes rather than inline styles: the card markup is stored once per molecule in the point
   data, so every byte of it is paid 33,008 times. */
.hc { font-family: 'IBM Plex Sans', sans-serif; width: min(460px, 100%); padding: 8px 10px; box-sizing: border-box;
      color: #1f2328; }
.hc-img { float: right; width: 120px; height: 120px; margin: 0 0 6px 12px; object-fit: contain; background: #fff;
          border: 1px solid #d0d7de; border-radius: 4px; }
.hc-name { font-size: 14px; font-weight: 600; line-height: 1.3; overflow-wrap: anywhere; }
.hc-facts { font-size: 11.5px; color: #57606a; margin-top: 3px; line-height: 1.5; }
.hc-desc { font-size: 12.5px; line-height: 1.45; margin-top: 8px; }
/* Context rows under the description: a fixed label column, so the same fact sits in the same place on every
   card and a viewer moving quickly over the map learns where to look. */
.hc-grid { display: grid; grid-template-columns: 88px 1fr; gap: 4px 10px; margin-top: 8px; font-size: 11.5px;
           line-height: 1.45; }
.hc-k { color: #57606a; font-weight: 600; font-size: 10px; text-transform: uppercase; letter-spacing: 0.04em;
        padding-top: 2px; }
/* Neighbour and partner names are clamped here, not in the data: PubChem titles are sometimes IUPAC names 200
   characters long, and the full name has to stay in the markup because the card is also the search text. */
.hc-v > span { display: inline-block; max-width: 220px; overflow: hidden; text-overflow: ellipsis;
               white-space: nowrap; vertical-align: bottom; }
.hc-iupac { font-size: 11px; color: #8b949e; margin-top: 6px; line-height: 1.4; overflow-wrap: anywhere;
            display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.hc-ids { font-size: 11px; color: #8b949e; margin-top: 2px; line-height: 1.5; }
.hc-hidden { display: none; }  /* search material with no visual job: the plain-text formula */
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
    assert (structure["cid"].to_numpy() == cids).all(), f"{files['structure'].name} is not aligned with corpus.parquet"
    return structure


def load_cross_labels(other_files: dict, cids: np.ndarray) -> pd.Series | None:
    """The other map's region of each molecule, from its layer with about CROSS_LAYER_TARGET regions."""
    if not other_files["labels"].exists():
        print(f"{other_files['labels'].name} does not exist yet: no cross colormap or hover line")
        return None
    labels = pd.read_parquet(other_files["labels"])
    assert (labels["cid"].to_numpy() == cids).all(), f"{other_files['labels'].name} is not aligned with corpus.parquet"
    cols = [c for c in labels.columns if c.startswith("label_layer_")]
    col = min(cols, key=lambda c: abs(labels[c].nunique() - config.CROSS_LAYER_TARGET))
    print(f"cross labels from {other_files['labels'].name} {col}: {labels[col].nunique() - 1} named")
    return labels[col].fillna("Unlabelled")


def load_rhea(cids: np.ndarray) -> pd.DataFrame:
    if not config.PATHS["rhea"].exists():
        raise SystemExit(f"{config.PATHS['rhea']} is missing: run stage 08 (`make rhea`) before rendering")
    rhea = pd.read_parquet(config.PATHS["rhea"])
    assert (rhea["cid"].to_numpy() == cids).all(), "rhea.parquet is not aligned with corpus.parquet"
    return rhea


def clamp(name: str) -> str:
    """A molecule name the card may truncate with an ellipsis (`.hc-v > span` in CUSTOM_CSS; a bare span because
    the markup is paid per molecule and a class name costs 0.06 MB compressed)."""
    return f"<span>{esc(name)}</span>"


def rhea_line(rhea: pd.DataFrame, names: dict) -> pd.Series:
    """One hover value per molecule in Rhea: its reaction count and its first partners, or "" outside Rhea."""

    def line(n_reactions, is_hub, partner_cids) -> str:
        if not n_reactions:
            return ""
        head = f"{n_reactions:,} reaction{'s' if n_reactions != 1 else ''}"
        if is_hub:
            return head + " (cofactor hub)"
        shown = list(partner_cids[: config.RHEA_PARTNERS_SHOWN])
        if not shown:
            return head
        more = len(partner_cids) - len(shown)
        return (
            head + " · partners: " + " · ".join(clamp(names[c]) for c in shown) + (f" (+{more} more)" if more else "")
        )

    return pd.Series([line(n, h, p) for n, h, p in zip(rhea["n_reactions"], rhea["is_hub"], rhea["partner_cids"])])


def region_line(*layers: pd.Series | np.ndarray) -> pd.Series:
    """A region name as a hover value: the first layer (finest first) that names the molecule, "" where none
    does. A quarter of molecules sit in no cluster at the finest layer but nearly all are inside a coarser one."""
    cols = [np.asarray(layer, dtype=object) for layer in layers]

    def pick(i) -> str:
        return next((esc(c[i]) for c in cols if c[i] and c[i] != "Unlabelled"), "")

    return pd.Series([pick(i) for i in range(len(cols[0]))])


def nearest_line(structure: pd.DataFrame, names: dict, min_similarity: float, show_similarity: bool) -> pd.Series:
    """One hover value per molecule naming its nearest neighbours in the other space, or "" where none is close
    enough. Tiny molecules have near-empty fingerprints and their nearest neighbours by structure are noise, so
    neighbours below the similarity floor are left out and the row disappears exactly where it would mislead.
    The similarity is shown only where it means something (`config.LAYOUTS[...]["neighbour_show_similarity"]`)."""

    def line(cids, sims) -> str:
        shown = zip(cids[: config.STRUCTURE_NEIGHBOURS_SHOWN], sims[: config.STRUCTURE_NEIGHBOURS_SHOWN])
        parts = [clamp(names[c]) + (f" ({s:.2f})" if show_similarity else "") for c, s in shown if s >= min_similarity]
        return " · ".join(parts)

    pairs = zip(structure["neighbour_cids"], structure["neighbour_similarity"])
    return pd.Series([line(c, s) for c, s in pairs])


def build_point_data(corpus: pd.DataFrame, rows: list[tuple[str, pd.Series | None]] = ()) -> pd.DataFrame:
    """One HTML column per molecule (`body`) that is both the hovercard and the search text, plus the CID.

    Storing the description once matters: the hover data ships as one gzipped JSON file, and a separate
    search column would double it. Search is a substring match over the column, so anything that should be
    searchable has to be in the markup, even if the card hides it (the plain-text formula sits in a hidden span
    because the visible one is marked up with subscripts). `rows` are (label, values) context rows rendered as
    a label/value grid under the description (the molecule's own region, the other map's region, the nearest
    molecules in the other space, the Rhea reaction context), "" where a molecule has none.
    """
    rows = [(label, vals.to_numpy()) for label, vals in rows if vals is not None]
    corpus = corpus.assign(context=[[(k, v[i]) for k, v in rows if v[i]] for i in range(len(corpus))])

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
        grid = "".join(f'<div class="hc-k">{k}</div><div class="hc-v">{v}</div>' for k, v in row["context"])
        iupac = row["pubchem_iUPACName"]
        formula = esc(row["pubchem_molecularFormula"])
        return (
            f'<div class="hc-name">{esc(row["name"])}</div>'
            f'<div class="hc-facts">{facts_line(row)}</div>'
            f'<div class="hc-desc">{esc(row["description"])}</div>'
            + (f'<div class="hc-grid">{grid}</div>' if grid else "")
            + (f'<div class="hc-iupac">{esc(iupac)}</div>' if isinstance(iupac, str) else "")
            + f'<div class="hc-ids">PubChem CID {row["cid"]} · ChEBI-20 {row["split"]} split'
            + (f'<span class="hc-hidden"> · {formula}</span>' if formula else "")
            + "</div>"
        )

    return pd.DataFrame({"cid": corpus["cid"].to_numpy(), "body": corpus.apply(body, axis=1).to_numpy()})


def add_open_graph(html_path, n_points: int, layout: str) -> None:
    og = {
        "og:title": config.map_title(layout),
        "og:description": (
            f"{n_points:,} molecules from the ChEBI-20 dataset, laid out by "
            f"{config.LAYOUTS[layout]['positioned_by']}. Pan, zoom, hover and search."
        ),
        "og:type": "website",
        "og:url": config.map_url(layout),
    }
    if (html_path.parent / "social-preview.png").exists():
        og["og:image"] = config.map_url(layout) + "social-preview.png"
        og["twitter:card"] = "summary_large_image"
    tags = "\n".join(f'<meta property="{k}" content="{html.escape(v, quote=True)}">' for k, v in og.items())
    text = html_path.read_text(encoding="utf-8")
    assert "<head>" in text, "no <head> in the rendered page"
    atomic_write_text(html_path, text.replace("<head>", "<head>\n" + tags, 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedding", default=config.EMBED_MODEL_KEY, choices=sorted(config.EMBED_MODELS))
    ap.add_argument("--layout", default="text", choices=sorted(config.LAYOUTS))
    args = ap.parse_args()
    layout, other = args.layout, config.other_layout(args.layout)
    files = config.keyed_files(args.embedding, layout)
    other_files = config.keyed_files(args.embedding, other)
    spec = config.LAYOUTS[layout]
    out_dir = files["map_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_parquet(config.PATHS["corpus"])
    cids = corpus["cid"].to_numpy()
    lay = np.load(files["umap"], allow_pickle=True)
    assert (lay["cid"] == cids).all(), f"{files['umap'].name} is not aligned with corpus.parquet"
    coords = lay["coords"]
    label_layers = load_label_layers(files, cids)
    labels_meta = json.loads(files["labels_meta"].read_text())
    placeholder = labels_meta["namer_model"].startswith("placeholder")
    if placeholder:
        print("labels are placeholders (stage 04 --preview); rendering an unnamed preview")
    structure = load_structure(files, cids)
    cross = load_cross_labels(other_files, cids)
    # What the hover and cross colormap call things depends on which map this is. Colormap names are short
    # noun phrases: the same string is the dropdown entry and the rotated colorbar title, and the qualifiers
    # ("first stated" excepted, since it says what the field is) live in the README.
    if layout == "text":
        nearest_label, cross_label, cross_other = "Nearest by structure", "Structural family", "Other family"
        coherence_name = "Structural coherence"
    else:
        nearest_label, cross_label, cross_other = "Nearest by description", "Description-map region", "Other region"
        coherence_name = "Description coherence"
    coherence_meta = {"field": "coherence", "description": coherence_name, "kind": "continuous", "cmap": "plasma"}
    rhea = load_rhea(cids)
    names = dict(zip(cids, corpus["name"]))
    extra = build_point_data(
        corpus,
        [
            ("Region", region_line(*label_layers)),
            (cross_label, region_line(cross) if cross is not None else None),
            (
                nearest_label,
                nearest_line(structure, names, spec["neighbour_min_similarity"], spec["neighbour_show_similarity"]),
            ),
            ("Rhea", rhea_line(rhea, names)),
        ],
    )

    organism_meta, organism_vals = categorical(
        "organism",
        "Metabolite organism (first stated)",
        top_n(corpus["primary_organism"], config.ORGANISM_TOP_N, "Other organism", "None stated"),
        neutral="None stated",
    )
    role_meta, role_vals = categorical(
        "role",
        "Biological role (first stated)",
        top_n(corpus["primary_role"], config.ROLE_TOP_N, "Other role", "None stated"),
        neutral="None stated",
    )
    # Stage 08: the commonest enzyme class among the Rhea reactions a molecule takes part in. Two greys: outside
    # Rhea, and in Rhea but in reactions without an EC number (38% of them), so the colours mean enzyme classes.
    ec_raw = rhea["ec_class"].to_numpy()
    ec_meta, ec_vals = categorical(
        "ec_class",
        "Enzyme class (Rhea)",
        np.where(ec_raw == "", "Not in Rhea", np.where(ec_raw == "Unassigned", "In Rhea, no EC class", ec_raw)),
        neutral="Not in Rhea",
    )
    ec_meta["color_mapping"]["In Rhea, no EC class"] = "#8c8c8c"
    charge_vals = corpus["pubchem_charge"].map(charge_bucket).to_numpy()
    charge_meta = {
        "field": "charge",
        "description": "Formal charge",
        "kind": "categorical",
        "color_mapping": {k: v for k, v in CHARGE_BUCKETS if k in set(charge_vals.tolist())},
        "show_legend": True,
    }
    split_meta, split_vals = categorical("split", "Dataset split", corpus["split"].to_numpy())
    mw = corpus["pubchem_molecularWeight"].astype(float)
    mw_vals = np.log10(mw.fillna(mw.median()).clip(lower=1)).to_numpy()
    mw_meta = {"field": "log_mw", "description": "Molecular weight (log10 Da)", "kind": "continuous", "cmap": "viridis"}
    xlogp = corpus["pubchem_xLogP"].astype(float)
    xlogp_vals = xlogp.fillna(xlogp.median()).clip(-10, 15).to_numpy()
    xlogp_meta = {
        "field": "xlogp",
        "description": "XLogP",
        "kind": "continuous",
        "cmap": "cividis",
    }
    # Stage 07: how far this map's neighbourhoods agree with the other space; see config.COHERENCE_MIN_SPAN.
    # DataMapPlot's colorbar treats a range whose two ends are both integers as integer data and rounds every
    # tick, so a 0-to-1 score gets ticks reading "0 0 1 1 1"; pulling the top just under 1 keeps 0.25 steps.
    coherence_vals = np.minimum(structure["coherence"].to_numpy(dtype=float), 0.9995)
    rawdata = [organism_vals, role_vals, ec_vals]
    metadata = [organism_meta, role_meta, ec_meta]
    if cross is not None:
        # The other map's regions coloured onto this one: a region that this map scatters shows up sprayed about.
        cross_meta, cross_vals = categorical(
            "cross",
            cross_label,
            top_n(cross.replace("Unlabelled", ""), config.CROSS_LEGEND_TOP_N, cross_other, "Unlabelled"),
            neutral="Unlabelled",
        )
        rawdata.append(cross_vals)
        metadata.append(cross_meta)
    rawdata += [charge_vals, mw_vals, xlogp_vals, coherence_vals, split_vals]
    metadata += [charge_meta, mw_meta, xlogp_meta, coherence_meta, split_meta]

    link, link_text = (
        ("morgan/", "the same molecules by structure")
        if layout == "text"
        else ("../", "the same molecules by description")
    )
    plot = datamapplot.create_interactive_plot(
        coords,
        *label_layers,
        hover_text=corpus["name"].tolist(),
        hover_text_html_template=HOVER_TEMPLATE,
        extra_point_data=extra,
        on_click=ON_CLICK,
        enable_search=True,
        search_field="body",
        title=config.map_title(layout),
        sub_title=(
            f"{len(corpus):,} molecules positioned by {spec['positioned_by']} · region names by Claude · "
            f'<a href="{link}" style="color:inherit;">{link_text}</a> · '
            f'<a href="{config.REPO_URL}" style="color:inherit;">code on GitHub</a>'
            + ("" if args.embedding == config.EMBED_MODEL_KEY else f" · exploration build: {args.embedding}")
            + (" · UNNAMED PREVIEW" if placeholder else "")
        ),
        font_family="IBM Plex Sans",
        cvd_safer=True,
        noise_label="Unlabelled",
        initial_zoom_fraction=config.MAP_INITIAL_ZOOM_FRACTION,
        colormap_rawdata=rawdata,
        colormap_metadata=metadata,
        custom_css=CUSTOM_CSS,
        custom_js=CUSTOM_JS,
        inline_data=False,
        offline_data_path=str(out_dir / config.MAP_DATA_PREFIX),
    )
    html_path = out_dir / "index.html"
    plot.save(str(html_path))
    add_open_graph(html_path, len(corpus), layout)

    print("output sizes:")
    total = 0
    for p in sorted(out_dir.iterdir()):
        if p.is_file() and (p.name == "index.html" or p.name.startswith(config.MAP_DATA_PREFIX + "_")):
            total += p.stat().st_size
            print(f"  {p.stat().st_size / 1e6:8.2f} MB  {p.name}")
    print(f"  {total / 1e6:8.2f} MB  total")
    n_b64 = len(re.findall(r"base64,", html_path.read_text(encoding="utf-8")))
    print(f"base64 blobs still inlined in index.html: {n_b64}")
    rel = html_path.relative_to(config.DOCS_DIR).parent if html_path.is_relative_to(config.DOCS_DIR) else ""
    print(f"wrote {html_path}; serve with `make serve` and open http://127.0.0.1:8765/{rel}")


if __name__ == "__main__":
    main()
