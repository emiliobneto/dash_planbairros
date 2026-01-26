# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from unicodedata import normalize as _ud_norm
import math
import re
import json
import base64
from decimal import Decimal

import pandas as pd
import streamlit as st

# Geo libs
try:
    import geopandas as gpd  # type: ignore
    import folium  # type: ignore
    from folium import Element  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
except Exception:
    gpd = None  # type: ignore
    folium = None  # type: ignore
    Element = None  # type: ignore
    st_folium = None  # type: ignore


# =============================================================================
# Config da página e identidade
# =============================================================================
st.set_page_config(
    page_title="PlanBairros",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PB_COLORS = {
    "amarelo": "#F4DD63",
    "verde": "#B1BF7C",
    "laranja": "#D58243",
    "telha": "#C65534",
    "teal": "#6FA097",
    "navy": "#14407D",
}
PB_NAVY = PB_COLORS["navy"]

# Jenks: 6 classes (discreto)
ORANGE_RED_GRAD = ["#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#e34a33", "#b30000"]

SIMPLIFY_TOL = 0.0006  # leve simplificação p/ performance


# =============================================================================
# Paths (GitHub) e logo
# =============================================================================
try:
    REPO_ROOT = Path(__file__).resolve().parent
except NameError:
    REPO_ROOT = Path.cwd()

DATA_DIR = REPO_ROOT / "limites_administrativos"
LOGO_PATH = REPO_ROOT / "assets" / "logo_todos.jpg"
LOGO_HEIGHT = 46  # px (ajuste fino de altura)


def _logo_data_uri() -> str:
    """
    Retorna data URI do logo local (assets/logo_todos.jpg).
    Se não existir, usa fallback online.
    """
    if LOGO_PATH.exists():
        suf = LOGO_PATH.suffix.lstrip(".").lower()
        mime = "jpeg" if suf in ("jpg", "jpeg") else suf
        b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
        return f"data:image/{mime};base64,{b64}"
    return (
        "https://raw.githubusercontent.com/streamlit/brand/refs/heads/main/"
        "logomark/streamlit-mark-color.png"
    )


# =============================================================================
# CSS (identidade + tooltips + header com logo)
# =============================================================================
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700;900&display=swap');
        html, body, .stApp {{
            font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
        }}
        .main .block-container {{
            padding-top: .15rem !important;
            padding-bottom: .6rem !important;
        }}
        .pb-row {{
            display:flex;
            align-items:center;
            gap:12px;
            margin-bottom:0;
        }}
        .pb-logo {{
            height:{LOGO_HEIGHT}px;
            width:auto;
            display:block;
            border-radius: 8px;
        }}
        .pb-header {{
            background:{PB_NAVY};
            color:#fff;
            border-radius:14px;
            padding:14px 15px;
            width:100%;
        }}
        .pb-title {{
            font-size:2.25rem;
            font-weight:900;
            line-height:1.05;
            letter-spacing:.2px;
        }}
        .pb-subtitle {{
            font-size:1.05rem;
            opacity:.95;
            margin-top:5px;
        }}
        .pb-card {{
            background:#fff;
            border:1px solid rgba(20,64,125,.10);
            box-shadow:0 1px 2px rgba(0,0,0,.04);
            border-radius:14px;
            padding:9px;
        }}

        /* Tooltips grandes Leaflet */
        .leaflet-tooltip.pb-big-tooltip,
        .leaflet-tooltip.pb-big-tooltip * {{
            font-size: 26px !important;
            font-weight: 900 !important;
            color: #111 !important;
            line-height: 1 !important;
        }}
        .leaflet-tooltip.pb-big-tooltip {{
            background:#fff !important;
            border: 2px solid #222 !important;
            border-radius: 10px !important;
            padding: 10px 14px !important;
            white-space: nowrap !important;
            pointer-events: none !important;
            box-shadow:0 2px 6px rgba(0,0,0,.2) !important;
            z-index: 200000 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Utilidades
# =============================================================================
def _slug(s: str) -> str:
    s2 = _ud_norm("NFKD", str(s)).encode("ASCII", "ignore").decode("ASCII")
    return re.sub(r"[^a-z0-9]+", "", s2.strip().lower())


def _find_file(folder: Path, candidates: List[str], exts: Tuple[str, ...]) -> Optional[Path]:
    if not folder.exists():
        return None
    wanted = {_slug(c) for c in candidates}
    for fp in folder.iterdir():
        if fp.is_file() and fp.suffix.lower() in exts and _slug(fp.stem) in wanted:
            return fp
    return None


def find_col(df_cols, *cands) -> Optional[str]:
    low = {c.lower(): c for c in df_cols}
    norm = {re.sub(r"[^a-z0-9]", "", k.lower()): v for k, v in low.items()}
    for c in cands:
        if not c:
            continue
        if c.lower() in low:
            return low[c.lower()]
        key = re.sub(r"[^a-z0-9]", "", c.lower())
        if key in norm:
            return norm[key]
    return None


def center_from_bounds(gdf) -> tuple[float, float]:
    minx, miny, maxx, maxy = gdf.total_bounds
    return ((miny + maxy) / 2, (minx + maxx) / 2)


def _to_float(x):
    if isinstance(x, Decimal):
        return float(x)
    return x


def to_float_series(s: pd.Series) -> pd.Series:
    if s.dtype == "object":
        return s.apply(_to_float).astype("Float64")
    return pd.to_numeric(s, errors="coerce").astype("Float64")


# =============================================================================
# Leitura e saneamento geoespacial (corrige KeyError do bounds)
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def read_gdf_parquet(path: Path) -> Optional["gpd.GeoDataFrame"]:
    if gpd is None:
        return None
    try:
        gdf = gpd.read_parquet(path)
    except Exception:
        return None

    try:
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        gdf = gdf.to_crs(4326)
    except Exception:
        try:
            gdf = gdf.set_crs(4326, allow_override=True)
        except Exception:
            pass
    return gdf


@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def read_gdf_geojson(path: Path) -> Optional["gpd.GeoDataFrame"]:
    if gpd is None:
        return None
    try:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        gdf = gdf.to_crs(4326)
        return gdf
    except Exception:
        return None


def _drop_bad_geoms(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty:
        return gdf
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notna()]
    try:
        gdf = gdf[~gdf.geometry.is_empty]
    except Exception:
        pass
    try:
        inv = ~gdf.geometry.is_valid
        if inv.any():
            gdf.loc[inv, "geometry"] = gdf.loc[inv, "geometry"].buffer(0)
    except Exception:
        pass
    gdf = gdf[gdf.geometry.notna()]
    try:
        gdf = gdf[~gdf.geometry.is_empty]
    except Exception:
        pass
    return gdf


def _simplify_safe(gdf: "gpd.GeoDataFrame", tol: float) -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty:
        return gdf
    gdf = gdf.copy()
    try:
        gdf["geometry"] = gdf.geometry.simplify(tol, preserve_topology=True)
        gdf = _drop_bad_geoms(gdf)
    except Exception:
        pass
    return gdf


def gdf_to_featurecollection(gdf: "gpd.GeoDataFrame", keep_cols: Optional[List[str]] = None) -> Optional[dict]:
    if gdf is None or gdf.empty:
        return None

    gdf = _drop_bad_geoms(gdf)
    if gdf.empty:
        return None

    if keep_cols is not None:
        cols = [c for c in keep_cols if c in gdf.columns]
        if "geometry" not in cols:
            cols = cols + ["geometry"]
        gdf = gdf[cols].copy()

    try:
        gj = json.loads(gdf.to_json())
    except Exception:
        return None

    feats = []
    for f in gj.get("features", []):
        geom = f.get("geometry")
        if not geom:
            continue
        if "coordinates" not in geom:
            continue
        feats.append(f)

    if not feats:
        return None

    gj["features"] = feats
    return gj


# =============================================================================
# Carregadores (paths do GitHub conforme informado)
# =============================================================================
def p_distritos() -> Optional[Path]:
    return _find_file(DATA_DIR, ["Distritos"], (".parquet",))


def p_subprefeitura() -> Optional[Path]:
    return _find_file(DATA_DIR, ["Subprefeitura", "subprefeitura"], (".parquet",))


def p_zonasod() -> Optional[Path]:
    return _find_file(DATA_DIR, ["ZonasOD2023", "ZonasOD"], (".parquet",))


def p_isocronas() -> Optional[Path]:
    return _find_file(DATA_DIR, ["isocronas", "isócronas"], (".parquet",))


def p_idcenso() -> Optional[Path]:
    return _find_file(DATA_DIR, ["IDCenso2023", "IDCenso"], (".parquet",))


def p_setores_vars() -> Optional[Path]:
    return _find_file(DATA_DIR, ["SetoresCensitarios2023", "SetoresCensitarios"], (".parquet",))


def p_area_verde() -> Optional[Path]:
    return _find_file(DATA_DIR, ["area_verde"], (".geojson",))


def p_rios() -> Optional[Path]:
    return _find_file(DATA_DIR, ["rios"], (".geojson",))


def p_linhas_metro() -> Optional[Path]:
    return _find_file(DATA_DIR, ["linhas_metro"], (".geojson",))


def p_linhas_trem() -> Optional[Path]:
    return _find_file(DATA_DIR, ["linhas_trem"], (".geojson",))


def load_admin(name: str) -> Optional["gpd.GeoDataFrame"]:
    if name == "Distritos":
        p = p_distritos()
    elif name == "Subprefeitura":
        p = p_subprefeitura()
    elif name == "ZonasOD2023":
        p = p_zonasod()
    elif name == "Isócronas":
        p = p_isocronas()
    else:
        p = None
    if not p:
        return None
    gdf = read_gdf_parquet(p)
    if gdf is None:
        return None
    gdf = _drop_bad_geoms(gdf)
    return _simplify_safe(gdf, SIMPLIFY_TOL)


def load_green_areas() -> Optional["gpd.GeoDataFrame"]:
    p = p_area_verde()
    if not p:
        return None
    gdf = read_gdf_geojson(p)
    if gdf is None:
        return None
    gdf = _drop_bad_geoms(gdf)
    return _simplify_safe(gdf, SIMPLIFY_TOL)


def load_rios() -> Optional["gpd.GeoDataFrame"]:
    p = p_rios()
    if not p:
        return None
    gdf = read_gdf_geojson(p)
    if gdf is None:
        return None
    gdf = _drop_bad_geoms(gdf)
    return _simplify_safe(gdf, SIMPLIFY_TOL)


def load_linhas_metro() -> Optional["gpd.GeoDataFrame"]:
    p = p_linhas_metro()
    if not p:
        return None
    gdf = read_gdf_geojson(p)
    if gdf is None:
        return None
    gdf = _drop_bad_geoms(gdf)
    return _simplify_safe(gdf, SIMPLIFY_TOL)


def load_linhas_trem() -> Optional["gpd.GeoDataFrame"]:
    p = p_linhas_trem()
    if not p:
        return None
    gdf = read_gdf_geojson(p)
    if gdf is None:
        return None
    gdf = _drop_bad_geoms(gdf)
    return _simplify_safe(gdf, SIMPLIFY_TOL)


def load_idcenso_geom() -> Optional["gpd.GeoDataFrame"]:
    p = p_idcenso()
    if not p:
        return None
    gdf = read_gdf_parquet(p)
    if gdf is None:
        return None
    gdf = _drop_bad_geoms(gdf)
    return _simplify_safe(gdf, SIMPLIFY_TOL)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def read_setores_vars_df(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def build_setores_joined_by_fid() -> Optional["gpd.GeoDataFrame"]:
    """
    IDCenso2023 -> geometria (fid + geometry)
    SetoresCensitarios2023 -> atributos (fid + variáveis)
    Join por fid.
    """
    if gpd is None:
        return None

    g_id = load_idcenso_geom()
    p_vars = p_setores_vars()

    if g_id is None or g_id.empty or not p_vars:
        return None

    df_vars = read_setores_vars_df(p_vars)
    if df_vars is None or df_vars.empty:
        return None

    fid_geom = find_col(g_id.columns, "fid", "FID")
    fid_vars = find_col(df_vars.columns, "fid", "FID")
    if not fid_geom or not fid_vars:
        return None

    g = g_id.copy()
    v = df_vars.copy()

    g[fid_geom] = pd.to_numeric(g[fid_geom].apply(_to_float), errors="coerce").astype("Int64")
    v[fid_vars] = pd.to_numeric(v[fid_vars].apply(_to_float), errors="coerce").astype("Int64")

    v = v.dropna(subset=[fid_vars]).drop_duplicates(subset=[fid_vars]).copy()

    if fid_vars != "fid":
        v = v.rename(columns={fid_vars: "fid"})
    else:
        v = v.rename(columns={fid_vars: "fid"})

    if fid_geom != "fid":
        g = g.rename(columns={fid_geom: "fid"})

    if "geometry" in v.columns:
        v = v.drop(columns=["geometry"])

    joined = g.merge(v, on="fid", how="left")
    joined = _drop_bad_geoms(joined)
    return joined


# =============================================================================
# Jenks (Natural Breaks) – 6 classes
# =============================================================================
def jenks_breaks(values: List[float], k: int) -> Optional[List[float]]:
    vals = [float(x) for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    vals = sorted(vals)
    n = len(vals)
    if n == 0:
        return None
    uniq = sorted(set(vals))
    if len(uniq) <= k:
        br = [uniq[0]] + uniq[1:] + [uniq[-1]]
        while len(br) < k + 1:
            br.insert(-1, br[-2])
        return br[: k + 1]

    mat1 = [[0] * (k + 1) for _ in range(n + 1)]
    mat2 = [[0] * (k + 1) for _ in range(n + 1)]

    for i in range(1, k + 1):
        mat1[1][i] = 1
        mat2[1][i] = 0
        for j in range(2, n + 1):
            mat2[j][i] = float("inf")

    v = 0.0
    for l in range(2, n + 1):
        s1 = s2 = w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = vals[i3 - 1]
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, k + 1):
                    if mat2[l][j] >= (v + mat2[i4][j - 1]):
                        mat1[l][j] = i3
                        mat2[l][j] = v + mat2[i4][j - 1]
        mat1[l][1] = 1
        mat2[l][1] = v

    breaks = [0.0] * (k + 1)
    breaks[k] = vals[-1]
    count = k
    idx = n
    while count >= 2:
        idxt = int(mat1[idx][count]) - 2
        breaks[count - 1] = vals[idxt]
        idx = int(mat1[idx][count]) - 1
        count -= 1
    breaks[0] = vals[0]
    return breaks


def jenks_class(v: float, breaks: List[float]) -> int:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return -1
    for i in range(1, len(breaks)):
        if v <= breaks[i]:
            return i - 1
    return len(breaks) - 2


# =============================================================================
# Folium – base Carto + panes
# =============================================================================
def make_carto_map(center=(-23.55, -46.63), zoom=11):
    if folium is None:
        return None
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)

    folium.TileLayer(
        tiles="CartoDB positron",
        name="Carto Positron",
        overlay=False,
        control=False,
    ).add_to(m)

    try:
        folium.map.CustomPane("admin", z_index=610).add_to(m)       # limites (linhas)
        folium.map.CustomPane("choropleth", z_index=620).add_to(m)  # setores (fill)
        folium.map.CustomPane("hydro", z_index=630).add_to(m)       # rios
        folium.map.CustomPane("rails", z_index=640).add_to(m)       # trem/metro
        folium.map.CustomPane("green", z_index=650).add_to(m)       # áreas verdes (acima)
    except Exception:
        pass

    return m


def add_admin_outline(m, gdf, name: str, color="#000000", weight=1.2, show=True):
    if folium is None or gdf is None or gdf.empty:
        return

    line = gdf[["geometry"]].copy()
    try:
        line["geometry"] = line.geometry.boundary
    except Exception:
        return

    line = _drop_bad_geoms(line)
    try:
        line = line[line.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    except Exception:
        pass
    line = _simplify_safe(line, SIMPLIFY_TOL)

    gj = gdf_to_featurecollection(line, keep_cols=["geometry"])
    if not gj:
        return

    fg = folium.FeatureGroup(name=f"{name}", show=show)
    folium.GeoJson(
        data=gj,
        pane="admin",
        style_function=lambda f: {"fillOpacity": 0, "color": color, "weight": weight},
    ).add_to(fg)
    fg.add_to(m)


def add_green(m, gdf, show=True):
    if folium is None or gdf is None or gdf.empty:
        return
    poly = _simplify_safe(_drop_bad_geoms(gdf[["geometry"]].copy()), SIMPLIFY_TOL)
    gj = gdf_to_featurecollection(poly, keep_cols=["geometry"])
    if not gj:
        return

    fg = folium.FeatureGroup(name="Áreas verdes", show=show)
    folium.GeoJson(
        data=gj,
        pane="green",
        style_function=lambda f: {
            "color": PB_COLORS["verde"], "weight": 0.0,
            "fillOpacity": 1.0, "fillColor": PB_COLORS["verde"]
        },
    ).add_to(fg)
    fg.add_to(m)


def add_lines(m, gdf, name: str, color: str, weight: float, pane: str, show=True):
    if folium is None or gdf is None or gdf.empty:
        return
    ln = _simplify_safe(_drop_bad_geoms(gdf[["geometry"]].copy()), SIMPLIFY_TOL)
    gj = gdf_to_featurecollection(ln, keep_cols=["geometry"])
    if not gj:
        return

    fg = folium.FeatureGroup(name=name, show=show)
    folium add_layer = folium.GeoJson(
        data=gj,
        pane=pane,
        style_function=lambda f: {"color": color, "weight": weight, "fillOpacity": 0},
    )
    add_layer.add_to(fg)
    fg.add_to(m)


# =============================================================================
# Pintura – setores (Jenks 6 classes) + Cluster
# =============================================================================
def paint_setores_jenks(m, setores: "gpd.GeoDataFrame", value_col: str, label: str):
    if folium is None or setores is None or setores.empty:
        return

    s = to_float_series(setores[value_col])
    vals = s.dropna().astype(float).tolist()
    br = jenks_breaks(vals, 6)
    if not br:
        st.info(f"Não foi possível calcular Jenks para '{label}'.")
        return

    df = setores[["geometry"]].copy()
    df["__v__"] = s
    df = _simplify_safe(_drop_bad_geoms(df), SIMPLIFY_TOL)
    if df.empty:
        return

    df["__k__"] = df["__v__"].apply(lambda x: jenks_class(float(x), br) if pd.notna(x) else -1).astype("Int64")
    gj = gdf_to_featurecollection(df, keep_cols=["geometry", "__k__", "__v__"])
    if not gj:
        return

    def style_fn(feat):
        k = feat["properties"].get("__k__", -1)
        try:
            k = int(k)
        except Exception:
            k = -1
        fill = "#c8c8c8" if k < 0 else ORANGE_RED_GRAD[min(k, len(ORANGE_RED_GRAD)-1)]
        return {"fillOpacity": 0.82, "weight": 0.0, "color": "#00000000", "fillColor": fill}

    fg = folium.FeatureGroup(name=label, show=True)
    folium.GeoJson(
        data=gj,
        pane="choropleth",
        style_function=style_fn,
        tooltip=folium.features.GeoJsonTooltip(
            fields=["__v__"], aliases=[label + ": "], sticky=True,
            labels=False, class_name="pb-big-tooltip",
        ),
        name=label,
    ).add_to(fg)
    fg.add_to(m)


def paint_setores_cluster(m, setores: "gpd.GeoDataFrame", cluster_col: str):
    if folium is None or setores is None or setores.empty:
        return

    color_map = {0:"#bf7db2", 1:"#f7bd6a", 2:"#cf651f", 3:"#ede4e6", 4:"#793393"}
    default_color = "#c8c8c8"

    df = setores[["geometry"]].copy()
    c = setores[cluster_col].apply(_to_float)
    df["__c__"] = pd.to_numeric(c, errors="coerce").astype("Int64")

    df = _simplify_safe(_drop_bad_geoms(df), SIMPLIFY_TOL)
    if df.empty:
        return

    gj = gdf_to_featurecollection(df, keep_cols=["geometry", "__c__"])
    if not gj:
        return

    def style_fn(feat):
        v = feat["properties"].get("__c__")
        try:
            v = int(v)
        except Exception:
            v = -1
        col = color_map.get(v, default_color)
        return {"fillOpacity": 0.78, "weight": 0.0, "color": "#00000000", "fillColor": col}

    fg = folium.FeatureGroup(name="Cluster", show=True)
    folium.GeoJson(
        data=gj,
        pane="choropleth",
        style_function=style_fn,
        name="Cluster",
    ).add_to(fg)
    fg.add_to(m)


# =============================================================================
# UI
# =============================================================================
def left_controls() -> Dict[str, Any]:
    st.markdown("### Variáveis (Setores Censitários)")
    var = st.selectbox(
        "Selecione a variável",
        [
            "Populacao",
            "Densidade_demografica",
            "Diferenca_elevacao",
            "elevacao",
            "raio_maximo_caminhada",
            "Cluster",
        ],
        index=1,
        key="pb_var",
        help="Atributos em SetoresCensitarios2023 aplicados na geometria do IDCenso2023 via join por fid.",
    )

    st.markdown("### Limites (linhas)")
    show_distritos = st.checkbox("Distritos", value=True)
    show_setores_outline = st.checkbox("IDCenso/Setores (contorno)", value=False)
    show_zonasod = st.checkbox("ZonasOD2023", value=False)
    show_subpref = st.checkbox("Subprefeitura", value=False)
    show_isocronas = st.checkbox("Isocronas (contorno)", value=False)

    st.markdown("### Referências (acima)")
    show_green = st.checkbox("Áreas verdes", value=True)
    show_rios = st.checkbox("Rios", value=True)
    show_metro = st.checkbox("Linhas de metrô", value=True)
    show_trem = st.checkbox("Linhas de trem", value=True)

    st.caption("Use o botão abaixo para limpar caches de dados.")
    if st.button("🧹 Limpar cache de dados", type="secondary"):
        st.cache_data.clear()
        st.success("Cache limpo. Recarregue as camadas.")

    return {
        "variavel": var,
        "show_distritos": show_distritos,
        "show_setores_outline": show_setores_outline,
        "show_zonasod": show_zonasod,
        "show_subpref": show_subpref,
        "show_isocronas": show_isocronas,
        "show_green": show_green,
        "show_rios": show_rios,
        "show_metro": show_metro,
        "show_trem": show_trem,
    }


# =============================================================================
# App
# =============================================================================
def main() -> None:
    if gpd is None or folium is None or st_folium is None:
        st.error("Este app requer `geopandas`, `folium` e `streamlit-folium` instalados.")
        return

    inject_css()

    # Header com LOGO (assets/logo_todos.jpg) dentro da barra azul
    logo_uri = _logo_data_uri()
    st.markdown(
        f"""
        <div class="pb-header">
          <div class="pb-row">
            <img src="{logo_uri}" class="pb-logo" />
            <div style="display:flex;flex-direction:column">
              <div class="pb-title">PlanBairros</div>
              <div class="pb-subtitle">Plataforma de visualização e planejamento em escala de bairro</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, map_col = st.columns([1, 4], gap="large")
    with left:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        ui = left_controls()
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Diagnóstico (paths / CRS / colunas)", expanded=False):
            st.write("REPO_ROOT:", str(REPO_ROOT))
            st.write("DATA_DIR:", str(DATA_DIR), "exists:", DATA_DIR.exists())
            st.write("LOGO_PATH:", str(LOGO_PATH), "exists:", LOGO_PATH.exists())
            st.write("Distritos:", str(p_distritos()))
            st.write("Subprefeitura:", str(p_subprefeitura()))
            st.write("ZonasOD2023:", str(p_zonasod()))
            st.write("Isocronas:", str(p_isocronas()))
            st.write("IDCenso2023:", str(p_idcenso()))
            st.write("SetoresCensitarios2023:", str(p_setores_vars()))
            st.write("area_verde.geojson:", str(p_area_verde()))
            st.write("rios.geojson:", str(p_rios()))
            st.write("linhas_metro.geojson:", str(p_linhas_metro()))
            st.write("linhas_trem.geojson:", str(p_linhas_trem()))

    with map_col:
        setores_join = build_setores_joined_by_fid()

        center = (-23.55, -46.63)
        g_d = load_admin("Distritos") if ui["show_distritos"] else None
        if g_d is not None and not g_d.empty:
            center = center_from_bounds(g_d)
        elif setores_join is not None and not setores_join.empty:
            center = center_from_bounds(setores_join)

        fmap = make_carto_map(center=center, zoom=11)

        # Limites administrativos (linhas)
        if ui["show_distritos"]:
            add_admin_outline(fmap, load_admin("Distritos"), "Distritos", color="#000000", weight=1.2, show=True)
        if ui["show_subpref"]:
            add_admin_outline(fmap, load_admin("Subprefeitura"), "Subprefeitura", color="#222222", weight=1.0, show=True)
        if ui["show_zonasod"]:
            add_admin_outline(fmap, load_admin("ZonasOD2023"), "ZonasOD2023", color="#333333", weight=0.9, show=True)
        if ui["show_isocronas"]:
            add_admin_outline(fmap, load_admin("Isócronas"), "Isocronas", color="#111111", weight=0.9, show=True)

        if ui["show_setores_outline"] and setores_join is not None and not setores_join.empty:
            add_admin_outline(fmap, setores_join, "IDCenso (contorno)", color="#000000", weight=0.8, show=True)

        # Variável por setores (join fid)
        if setores_join is None or setores_join.empty:
            st.warning("Não foi possível montar os Setores (join por fid). Verifique IDCenso2023/SetoresCensitarios2023 e a coluna fid.")
        else:
            var = ui["variavel"]

            col_pop = find_col(setores_join.columns, "Populacao", "populacao")
            col_den = find_col(setores_join.columns, "densidade_demografica", "nsidade_demograf", "densidade_demograf", "densidade")
            col_de = find_col(setores_join.columns, "Diferenca_elevacao", "diferenca_elevacao", "dif_elev", "dif_elevacao")
            col_el = find_col(setores_join.columns, "elevacao", "Elevacao")
            col_ra = find_col(setores_join.columns, "raio_maximo_caminhada", "raio_maximo", "raio_max")
            col_cl = find_col(setores_join.columns, "Cluster", "cluster")

            if var == "Populacao" and col_pop:
                paint_setores_jenks(fmap, setores_join, col_pop, "Populacao")
            elif var == "Densidade_demografica" and col_den:
                paint_setores_jenks(fmap, setores_join, col_den, "Densidade demografica")
            elif var == "Diferenca_elevacao" and col_de:
                paint_setores_jenks(fmap, setores_join, col_de, "Diferenca elevacao")
            elif var == "elevacao" and col_el:
                paint_setores_jenks(fmap, setores_join, col_el, "Elevacao")
            elif var == "raio_maximo_caminhada" and col_ra:
                paint_setores_jenks(fmap, setores_join, col_ra, "Raio maximo caminhada")
            elif var == "Cluster" and col_cl:
                paint_setores_cluster(fmap, setores_join, col_cl)
            else:
                st.info("Coluna da variável não encontrada. Verifique nomes no SetoresCensitarios2023.")

        # Overlays (acima)
        if ui["show_rios"]:
            add_lines(fmap, load_rios(), "Rios", color="#2b7bff", weight=2.0, pane="hydro", show=True)
        if ui["show_metro"]:
            add_lines(fmap, load_linhas_metro(), "Linhas de metrô", color="#000000", weight=2.6, pane="rails", show=True)
        if ui["show_trem"]:
            add_lines(fmap, load_linhas_trem(), "Linhas de trem", color="#000000", weight=2.6, pane="rails", show=True)
        if ui["show_green"]:
            add_green(fmap, load_green_areas(), show=True)

        # Layer control (inferior direito)
        try:
            folium.LayerControl(position="bottomright", collapsed=False).add_to(fmap)
        except Exception:
            pass

        st_folium(
            fmap,
            height=780,
            use_container_width=True,
            key="map_view",
        )


if __name__ == "__main__":
    main()
