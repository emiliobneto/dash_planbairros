# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from unicodedata import normalize as _ud_norm
import base64
import math
import re
from decimal import Decimal

import numpy as np
import pandas as pd
import streamlit as st

# Geo libs
try:
    import geopandas as gpd  # type: ignore
    import folium  # type: ignore
    from folium import Element  # type: ignore
except Exception:
    gpd = None  # type: ignore
    folium = None  # type: ignore
    Element = None  # type: ignore

# Para Jenks (aceleração com FisherJenksSampled)
try:
    import mapclassify  # type: ignore
except Exception:
    mapclassify = None  # type: ignore

# Render rápido (evita st_folium.get_bounds lento e KeyError)
try:
    from streamlit.components.v1 import html as st_html  # type: ignore
except Exception:
    st_html = None  # type: ignore


# =============================================================================
# Config da página e identidade visual
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

# Paleta (6 classes – Jenks)
# (usei uma sequência quente/contrastada, sem azul/verde para não “brigar” com rios/áreas verdes)
JENKS_6 = ["#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#e34a33"]

# Simplificações: ajuste fino para performance sem “sumir” feição
SIMPLIFY_CHORO = 0.0006   # choroplético (reduz bastante vértices)
SIMPLIFY_LINES = 0.0009   # linhas de setores/limites (mais leve)
SIMPLIFY_REF   = 0.0007   # overlays (rios/verde/trem/metro)

LOGO_HEIGHT = 44

# =============================================================================
# Paths
# =============================================================================
try:
    REPO_ROOT = Path(__file__).resolve().parent
except NameError:
    REPO_ROOT = Path.cwd()

DATA_DIR = (
    REPO_ROOT / "limites_administrativos"
    if (REPO_ROOT / "limites_administrativos").exists()
    else REPO_ROOT
)

LOGO_PATH = REPO_ROOT / "assets" / "logo_todos.jpg"


# =============================================================================
# CSS + Logo
# =============================================================================
def _logo_data_uri() -> str:
    if LOGO_PATH.exists():
        b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
        # jpg/png
        ext = LOGO_PATH.suffix.lstrip(".").lower()
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        return f"data:image/{mime};base64,{b64}"
    # fallback neutro
    return "https://raw.githubusercontent.com/streamlit/brand/refs/heads/main/logomark/streamlit-mark-color.png"


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700;900&display=swap');
        html, body, .stApp {{
            font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif !important;
        }}
        .main .block-container {{
            padding-top: .15rem !important;
            padding-bottom: .6rem !important;
        }}
        .pb-row {{
            display:flex; align-items:center; gap:12px; margin-bottom:0;
        }}
        .pb-logo {{
            height:{LOGO_HEIGHT}px; width:auto; display:block;
            border-radius: 10px;
        }}
        .pb-header {{
            background:{PB_NAVY}; color:#fff; border-radius:14px;
            padding:14px 15px; width:100%;
        }}
        .pb-title {{
            font-size:2.15rem; font-weight:900; line-height:1.05; letter-spacing:.2px;
        }}
        .pb-subtitle {{
            font-size:1.05rem; opacity:.95; margin-top:5px;
        }}
        .pb-card {{
            background:#fff;
            border:1px solid rgba(20,64,125,.10);
            box-shadow:0 1px 2px rgba(0,0,0,.04);
            border-radius:14px;
            padding:10px;
        }}

        /* Tooltip grande (Leaflet) */
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
            box-shadow: 0 2px 6px rgba(0,0,0,.2) !important;
            z-index: 200000 !important;
        }}

        /* Move LayerControl para canto inferior direito (mais “dashboard”) */
        .leaflet-control-layers {{
            border-radius: 10px !important;
            box-shadow: 0 2px 6px rgba(0,0,0,.18) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Utilidades: nomes e colunas
# =============================================================================
def _slug(s: str) -> str:
    s2 = _ud_norm("NFKD", str(s)).encode("ASCII", "ignore").decode("ASCII")
    return re.sub(r"[^a-z0-9]+", "", s2.strip().lower())


def _find_first_file(folder: Path, stems: List[str], exts: Tuple[str, ...] = ("parquet", "geojson", "json")) -> Optional[Path]:
    """Procura por arquivos cujo stem bata (slug) com algum candidato. Aceita múltiplas extensões."""
    if not folder.exists():
        return None
    wanted = {_slug(x) for x in stems}
    for ext in exts:
        for fp in folder.glob(f"*.{ext}"):
            if _slug(fp.stem) in wanted:
                return fp
    return None


def find_col(cols: List[str] | pd.Index, *cands: str) -> Optional[str]:
    cols = list(cols)
    low = {c.lower(): c for c in cols}
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


def to_float_series(s: pd.Series) -> pd.Series:
    if s.dtype == "object":
        def _coerce(x):
            if isinstance(x, Decimal):
                return float(x)
            return x
        s = s.apply(_coerce)
    return pd.to_numeric(s, errors="coerce").astype("Float64")


def _ensure_crs_4326(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf.to_crs(4326)


def _clean_geoms(gdf: "gpd.GeoDataFrame", allowed: Tuple[str, ...]) -> "gpd.GeoDataFrame":
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    # filtra tipos
    try:
        gdf = gdf[gdf.geometry.geom_type.isin(list(allowed))].copy()
    except Exception:
        pass
    return gdf


def center_from_bounds(gdf: "gpd.GeoDataFrame") -> Tuple[float, float]:
    minx, miny, maxx, maxy = gdf.total_bounds
    return ((miny + maxy) / 2.0, (minx + maxx) / 2.0)


# =============================================================================
# Leitura (cache)
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def _read_gdf_any(path: str) -> Optional["gpd.GeoDataFrame"]:
    """Lê parquet/geojson e padroniza CRS."""
    if gpd is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        if p.suffix.lower() in (".geojson", ".json"):
            gdf = gpd.read_file(p)
        else:
            gdf = gpd.read_parquet(p)
        gdf = _ensure_crs_4326(gdf)
        return gdf
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def _read_df_parquet(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def load_admin_layer(name: str) -> Optional["gpd.GeoDataFrame"]:
    stems = {
        "Distritos": ["Distritos"],
        "SetoresCensitarios2023": ["SetoresCensitarios2023", "IDCenso2023", "IDCenso"],
        "ZonasOD2023": ["ZonasOD2023", "ZonasOD"],
        "Subprefeitura": ["Subprefeitura", "Subprefeituras"],
        "Isócronas": ["isocronas", "isócronas", "Isocronas"],
    }.get(name, [name])

    fp = _find_first_file(DATA_DIR, stems)
    return _read_gdf_any(str(fp)) if fp else None


def _paths_setores_join() -> Tuple[Optional[Path], Optional[Path]]:
    """Geometria (IDCenso) + Métricas (SetoresCensitarios2023) – ambos por fid."""
    geom_fp = _find_first_file(DATA_DIR, ["IDCenso2023", "IDCenso", "SetoresCensitarios2023"])
    metrics_fp = _find_first_file(DATA_DIR, ["SetoresCensitarios2023", "SetoresCensitarios"])
    return geom_fp, metrics_fp


@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def build_setores_gdf(geom_path: str, metrics_path: str) -> Optional["gpd.GeoDataFrame"]:
    """Join por fid: geometria (IDCenso) + métricas (SetoresCensitarios2023)."""
    if gpd is None:
        return None

    ggeom = _read_gdf_any(geom_path)
    if ggeom is None or ggeom.empty:
        return None
    ggeom = _ensure_crs_4326(ggeom)
    ggeom = _clean_geoms(ggeom, ("Polygon", "MultiPolygon"))

    fid_g = find_col(ggeom.columns, "fid", "FID")
    if not fid_g:
        return None

    dfm = _read_df_parquet(metrics_path)
    if dfm is None or dfm.empty:
        # se não existir métricas, ainda devolve geometria
        ggeom = ggeom.rename(columns={fid_g: "fid"})
        ggeom["fid"] = pd.to_numeric(ggeom["fid"], errors="coerce").astype("Int64")
        return ggeom

    fid_m = find_col(dfm.columns, "fid", "FID")
    if not fid_m:
        return None

    # normaliza fid
    ggeom = ggeom.rename(columns={fid_g: "fid"}).copy()
    ggeom["fid"] = pd.to_numeric(ggeom["fid"], errors="coerce").astype("Int64")

    dfm = dfm.rename(columns={fid_m: "fid"}).copy()
    dfm["fid"] = pd.to_numeric(dfm["fid"], errors="coerce").astype("Int64")

    # remove geometry duplicada nas métricas (se existir)
    gcol = find_col(dfm.columns, "geometry", "geom")
    if gcol:
        dfm = dfm.drop(columns=[gcol])

    out = ggeom.merge(dfm, on="fid", how="left")
    return out


def load_reference_overlays() -> Dict[str, Optional["gpd.GeoDataFrame"]]:
    """Áreas verdes, rios, trem, metrô (aceita geojson/parquet)."""
    layers: Dict[str, Optional["gpd.GeoDataFrame"]] = {}

    # nomes vistos no print do diretório
    layers["areas_verdes"] = _read_gdf_any(str(_find_first_file(DATA_DIR, ["area_verde", "areas_verdes", "areaverde"]))) \
        if _find_first_file(DATA_DIR, ["area_verde", "areas_verdes", "areaverde"]) else None

    layers["rios"] = _read_gdf_any(str(_find_first_file(DATA_DIR, ["rios"]))) \
        if _find_first_file(DATA_DIR, ["rios"]) else None

    layers["metro"] = _read_gdf_any(str(_find_first_file(DATA_DIR, ["linhas metro", "linhas_metro", "metro"]))) \
        if _find_first_file(DATA_DIR, ["linhas metro", "linhas_metro", "metro"]) else None

    layers["trem"] = _read_gdf_any(str(_find_first_file(DATA_DIR, ["linhas trem", "linhas_trem", "trem"]))) \
        if _find_first_file(DATA_DIR, ["linhas trem", "linhas_trem", "trem"]) else None

    # padroniza CRS e tipos
    if layers["areas_verdes"] is not None and not layers["areas_verdes"].empty:
        layers["areas_verdes"] = _ensure_crs_4326(layers["areas_verdes"])
        layers["areas_verdes"] = _clean_geoms(layers["areas_verdes"], ("Polygon", "MultiPolygon"))
    if layers["rios"] is not None and not layers["rios"].empty:
        layers["rios"] = _ensure_crs_4326(layers["rios"])
        layers["rios"] = _clean_geoms(layers["rios"], ("LineString", "MultiLineString"))
    if layers["metro"] is not None and not layers["metro"].empty:
        layers["metro"] = _ensure_crs_4326(layers["metro"])
        layers["metro"] = _clean_geoms(layers["metro"], ("LineString", "MultiLineString"))
    if layers["trem"] is not None and not layers["trem"].empty:
        layers["trem"] = _ensure_crs_4326(layers["trem"])
        layers["trem"] = _clean_geoms(layers["trem"], ("LineString", "MultiLineString"))

    return layers


# =============================================================================
# Folium: base Carto + panes
# =============================================================================
def make_carto_map(center=(-23.55, -46.63), zoom=11):
    if folium is None:
        return None

    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)

    # basemap fixo (Carto Positron)
    folium.TileLayer(
        tiles="cartodbpositron",
        name="Carto Positron",
        overlay=False,
        control=False,
    ).add_to(m)

    # panes para ordem de desenho (z-index crescente)
    try:
        folium.map.CustomPane("choropleth", z_index=550).add_to(m)
        folium.map.CustomPane("admin", z_index=650).add_to(m)
        folium.map.CustomPane("sector_lines", z_index=660).add_to(m)
        folium.map.CustomPane("ref", z_index=700).add_to(m)
        folium.map.CustomPane("top", z_index=800).add_to(m)  # áreas verdes acima de tudo
    except Exception:
        pass

    # JS: joga layer control para bottomright (quando existir)
    if Element is not None:
        js = """
        <script>
          (function(){
            function moveControl(){
              var el = document.querySelector('.leaflet-control-layers');
              if(!el) return;
              var parent = el.parentElement;
              if(!parent) return;
              el.classList.add('leaflet-control-layers-expanded');
              // Leaflet não expõe API de posição do control depois de criado,
              // então só garantimos “visual” e deixamos no canto padrão.
            }
            setTimeout(moveControl, 350);
          })();
        </script>
        """
        m.get_root().html.add_child(Element(js))

    return m


def add_admin_outline(m, gdf, layer_name: str, color="#000000", weight=1.2, pane="admin"):
    if folium is None or gdf is None or gdf.empty:
        return

    gdf = _ensure_crs_4326(gdf)
    gdf = _clean_geoms(gdf, ("Polygon", "MultiPolygon"))

    line = gdf[["geometry"]].copy()
    line["geometry"] = line.geometry.boundary
    try:
        line["geometry"] = line.geometry.simplify(SIMPLIFY_LINES, preserve_topology=True)
    except Exception:
        pass

    line = _clean_geoms(line, ("LineString", "MultiLineString"))
    if line.empty:
        return

    folium.GeoJson(
        data=line.to_json(),
        name=f"{layer_name} (contorno)",
        pane=pane,
        style_function=lambda f: {"fillOpacity": 0, "color": color, "weight": weight, "opacity": 1.0},
        control=True,
        smooth_factor=0.0,
    ).add_to(m)


def add_sector_lines(m, setores_geom_gdf: "gpd.GeoDataFrame", color="#222222", weight=0.35, opacity=0.55):
    """
    (ALTERAÇÃO 1) Linhas dos Setores (IDCenso/SetoresCensitarios2023) como contorno.
    IMPORTANTE: isso pode ser pesado se renderizar a cidade inteira.
    Por isso: simplifica + manda só boundary.
    """
    if folium is None or setores_geom_gdf is None or setores_geom_gdf.empty:
        return
    gdf = setores_geom_gdf[["geometry"]].copy()
    gdf = _ensure_crs_4326(gdf)
    gdf = _clean_geoms(gdf, ("Polygon", "MultiPolygon"))

    gdf["geometry"] = gdf.geometry.boundary
    try:
        gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_LINES, preserve_topology=True)
    except Exception:
        pass

    gdf = _clean_geoms(gdf, ("LineString", "MultiLineString"))
    if gdf.empty:
        return

    folium.GeoJson(
        data=gdf.to_json(),
        name="Setores censitários (linhas)",
        pane="sector_lines",
        style_function=lambda f: {"fillOpacity": 0, "color": color, "weight": weight, "opacity": opacity},
        control=True,
        smooth_factor=0.0,
    ).add_to(m)


def add_reference_layers(m, ref: Dict[str, Optional["gpd.GeoDataFrame"]]):
    """Áreas verdes acima de tudo; rios azul; trem/metro preto."""
    if folium is None or m is None:
        return

    # Áreas verdes (top)
    g = ref.get("areas_verdes")
    if g is not None and not g.empty:
        gg = g[["geometry"]].copy()
        try:
            gg["geometry"] = gg.geometry.simplify(SIMPLIFY_REF, preserve_topology=True)
        except Exception:
            pass
        gg = _clean_geoms(gg, ("Polygon", "MultiPolygon"))
        if not gg.empty:
            folium.GeoJson(
                data=gg.to_json(),
                name="Áreas verdes",
                pane="top",
                style_function=lambda f: {"color": "#2E7D32", "weight": 0.6, "fillColor": "#2E7D32", "fillOpacity": 1.0},
                control=True,
            ).add_to(m)

    # Rios (ref)
    r = ref.get("rios")
    if r is not None and not r.empty:
        rr = r[["geometry"]].copy()
        try:
            rr["geometry"] = rr.geometry.simplify(SIMPLIFY_REF, preserve_topology=True)
        except Exception:
            pass
        rr = _clean_geoms(rr, ("LineString", "MultiLineString"))
        if not rr.empty:
            folium.GeoJson(
                data=rr.to_json(),
                name="Rios",
                pane="ref",
                style_function=lambda f: {"color": "#1E88E5", "weight": 1.2, "opacity": 1.0},
                control=True,
            ).add_to(m)

    # Metrô (ref)
    metro = ref.get("metro")
    if metro is not None and not metro.empty:
        mm = metro[["geometry"]].copy()
        try:
            mm["geometry"] = mm.geometry.simplify(SIMPLIFY_REF, preserve_topology=True)
        except Exception:
            pass
        mm = _clean_geoms(mm, ("LineString", "MultiLineString"))
        if not mm.empty:
            folium.GeoJson(
                data=mm.to_json(),
                name="Linhas de metrô",
                pane="ref",
                style_function=lambda f: {"color": "#000000", "weight": 2.0, "opacity": 1.0},
                control=True,
            ).add_to(m)

    # Trem (ref)
    trem = ref.get("trem")
    if trem is not None and not trem.empty:
        tt = trem[["geometry"]].copy()
        try:
            tt["geometry"] = tt.geometry.simplify(SIMPLIFY_REF, preserve_topology=True)
        except Exception:
            pass
        tt = _clean_geoms(tt, ("LineString", "MultiLineString"))
        if not tt.empty:
            folium.GeoJson(
                data=tt.to_json(),
                name="Linhas de trem",
                pane="ref",
                style_function=lambda f: {"color": "#000000", "weight": 2.0, "opacity": 1.0},
                control=True,
            ).add_to(m)


# =============================================================================
# Choroplético (Jenks 6 classes) – cache + otimizações
# =============================================================================
def _jenks_6_breaks(values: np.ndarray) -> Optional[List[float]]:
    if mapclassify is None:
        return None
    v = values[np.isfinite(values)]
    if v.size < 10:
        return None
    # sampled: MUITO mais rápido em n grande
    cls = mapclassify.FisherJenksSampled(v, k=6, pct=0.12, truncate=True)
    # bins são limites superiores por classe
    bins = list(map(float, cls.bins))
    return bins


@st.cache_data(show_spinner=False, ttl=3600, max_entries=24)
def prepare_choropleth_geojson(
    geom_path: str,
    metrics_path: str,
    value_col_wanted: str,
) -> Optional[Dict[str, Any]]:
    """
    Retorna:
      - geojson_str: string geojson com __v__, __k__, __color__
      - breaks: bins Jenks
      - vmin/vmax
    """
    setores = build_setores_gdf(geom_path, metrics_path)
    if setores is None or setores.empty:
        return None

    # coluna real
    val_col = find_col(setores.columns, value_col_wanted)
    if not val_col:
        # tenta candidatos comuns (do seu print)
        val_col = find_col(setores.columns, "Populacao", "densidade_demograf", "densidade_demografica",
                           "Diferenca_elevacao", "elevacao", "raio_maximo_caminhada")
    if not val_col:
        return None

    gdf = setores[["geometry", "fid", val_col]].copy()
    gdf = _ensure_crs_4326(gdf)
    gdf = _clean_geoms(gdf, ("Polygon", "MultiPolygon"))

    s = to_float_series(gdf[val_col])
    gdf["__v__"] = s

    vmin = float(np.nanmin(s.values)) if np.isfinite(np.nanmin(s.values)) else 0.0
    vmax = float(np.nanmax(s.values)) if np.isfinite(np.nanmax(s.values)) else 0.0

    # Jenks breaks (6 classes)
    breaks = _jenks_6_breaks(s.to_numpy(dtype=float))
    if breaks is None:
        # fallback: quantis (se mapclassify não estiver disponível)
        q = np.nanquantile(s.to_numpy(dtype=float), [1/6, 2/6, 3/6, 4/6, 5/6, 1.0])
        breaks = list(map(float, q))

    # classifica
    # bins define o teto de cada classe
    def classify(v: float) -> int:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return -1
        for i, b in enumerate(breaks):
            if v <= b:
                return i
        return 5

    gdf["__k__"] = gdf["__v__"].apply(lambda x: classify(float(x)) if pd.notna(x) else -1).astype("int16")
    gdf["__color__"] = gdf["__k__"].apply(lambda k: JENKS_6[k] if 0 <= k < 6 else "#c8c8c8")

    # simplifica MUITO para reduzir payload
    try:
        gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_CHORO, preserve_topology=True)
    except Exception:
        pass

    # reduz colunas antes de serializar
    out = gdf[["geometry", "__v__", "__k__", "__color__"]].copy()
    out = _clean_geoms(out, ("Polygon", "MultiPolygon"))
    if out.empty:
        return None

    geojson_str = out.to_json()

    return {
        "geojson_str": geojson_str,
        "breaks": breaks,
        "vmin": vmin,
        "vmax": vmax,
        "value_col": val_col,
    }


def add_choropleth_layer(m, payload: Dict[str, Any], label: str):
    if folium is None or payload is None:
        return

    geojson_str = payload["geojson_str"]

    def style_fn(feat):
        c = feat["properties"].get("__color__", "#c8c8c8")
        return {"fillOpacity": 0.80, "weight": 0.0, "color": "#00000000", "fillColor": c}

    folium.GeoJson(
        data=geojson_str,
        name=label,
        pane="choropleth",
        style_function=style_fn,
        tooltip=folium.features.GeoJsonTooltip(
            fields=["__v__"],
            aliases=[label + ": "],
            sticky=True,
            labels=False,
            class_name="pb-big-tooltip",
        ),
        control=True,
    ).add_to(m)


# =============================================================================
# UI
# =============================================================================
def left_controls() -> Dict[str, Any]:
    st.markdown("### Variáveis (Setores Censitários 2023)")
    var = st.selectbox(
        "Selecione a variável",
        [
            "— Selecione —",
            "População",
            "Densidade demográfica",
            "Diferença de elevação",
            "Elevação média",
            "Raio máximo de caminhada",
            "Cluster",
        ],
        index=0,
        key="pb_var",
        help="Choroplético é sempre em Setores/IDCenso (join por fid).",
    )

    st.markdown("### Limites administrativos")
    limite = st.selectbox(
        "Contorno",
        ["Distritos", "ZonasOD2023", "Subprefeitura", "Isócronas"],
        index=0,
        key="pb_limite",
        help="Exibe o contorno do limite selecionado.",
    )

    # (ALTERAÇÃO 1) Setores como linhas dentro dos limites
    show_sector_lines = st.checkbox(
        "Mostrar linhas dos setores (IDCenso/Setores 2023)",
        value=False,  # deixa leve por padrão
        key="pb_sector_lines",
        help="Desenha o contorno dos setores como layer adicional. Pode pesar.",
    )

    st.markdown("### Camadas de referência")
    show_refs = st.checkbox("Mostrar áreas verdes, rios, trem e metrô", value=True, key="pb_refs")

    st.caption("Cache ajuda muito nos choropléticos.")
    if st.button("🧹 Limpar cache", type="secondary"):
        st.cache_data.clear()
        st.success("Cache limpo.")

    return {
        "variavel": var,
        "limite": limite,
        "show_sector_lines": show_sector_lines,
        "show_refs": show_refs,
    }


# =============================================================================
# Render rápido do mapa (sem st_folium)
# =============================================================================
def render_map_fast(m, height: int = 780):
    if st_html is None:
        st.warning("streamlit.components não disponível. (deveria estar)")
        return
    html = m.get_root().render()
    st_html(html, height=height, scrolling=False)


# =============================================================================
# App principal
# =============================================================================
def main() -> None:
    if gpd is None or folium is None:
        st.error("Este app requer `geopandas` e `folium` instalados.")
        return

    inject_css()

    # Header com logo do projeto
    logo_uri = _logo_data_uri()
    st.markdown(
        f"""
        <div class="pb-header">
          <div class="pb-row">
            <img class="pb-logo" src="{logo_uri}" />
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

    # Carrega limites (normalmente leve)
    limite_gdf = load_admin_layer(ui["limite"])

    # Centro inicial (SP)
    center = (-23.55, -46.63)
    if limite_gdf is not None and not limite_gdf.empty:
        center = center_from_bounds(_clean_geoms(limite_gdf, ("Polygon", "MultiPolygon")))

    # Basemap Carto sempre
    m = make_carto_map(center=center, zoom=11)

    # Contorno do limite selecionado
    if limite_gdf is not None and not limite_gdf.empty:
        add_admin_outline(m, limite_gdf, ui["limite"], color="#000000", weight=1.2, pane="admin")
    else:
        st.info("Limite administrativo não encontrado em limites_administrativos/.")

    # Overlays de referência (verde/rios/metro/trem)
    if ui["show_refs"]:
        ref = load_reference_overlays()
        add_reference_layers(m, ref)

    # (ALTERAÇÃO 1) Linhas de setores junto com limites
    if ui["show_sector_lines"]:
        geom_fp, metrics_fp = _paths_setores_join()
        if geom_fp is None:
            st.warning("Não encontrei IDCenso/SetoresCensitarios2023 para desenhar linhas.")
        else:
            setores_geom = _read_gdf_any(str(geom_fp))
            if setores_geom is None or setores_geom.empty:
                st.warning("Falha ao ler o arquivo de geometria (IDCenso/Setores).")
            else:
                add_sector_lines(m, setores_geom)

    # Choroplético apenas quando variável for escolhida (deixa o início leve)
    var = ui["variavel"]
    if var != "— Selecione —":
        geom_fp, metrics_fp = _paths_setores_join()
        if geom_fp is None or metrics_fp is None:
            st.warning("Não encontrei IDCenso2023/SetoresCensitarios2023 para choroplético (join por fid).")
        else:
            wanted = {
                "População": "Populacao",
                "Densidade demográfica": "densidade_demograf",
                "Diferença de elevação": "Diferenca_elevacao",
                "Elevação média": "elevacao",
                "Raio máximo de caminhada": "raio_maximo_caminhada",
                "Cluster": "Cluster",
            }.get(var, var)

            payload = prepare_choropleth_geojson(str(geom_fp), str(metrics_fp), wanted)

            if payload is None:
                st.warning(f"Não consegui montar choroplético para '{var}'. Verifique nome da coluna e `fid`.")
            else:
                # Cluster não é choroplético Jenks aqui (por enquanto); você pode pintar categórico depois.
                if var == "Cluster":
                    # pinta como valores (rápido) só para não “sumir”
                    add_choropleth_layer(m, payload, "Cluster (numérico)")
                else:
                    add_choropleth_layer(m, payload, var)

    # LayerControl
    folium.LayerControl(collapsed=True, position="topright").add_to(m)

    with map_col:
        render_map_fast(m, height=820)


if __name__ == "__main__":
    main()
