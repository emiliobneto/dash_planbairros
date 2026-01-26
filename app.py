# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import base64
import math
import re
import json
import numpy as np
import pandas as pd
import streamlit as st

# ====================== imports dinâmicos ======================

def _import_stack():
    try:
        import geopandas as gpd
        import pydeck as pdk
        from pydeck.types import Function as DeckFunction  # noqa: F401
        from shapely import wkb, wkt
        from shapely.geometry import base as shapely_base
        from pyarrow import parquet as pq
        return gpd, pdk, DeckFunction, wkb, wkt, shapely_base, pq
    except ImportError as e:
        st.set_page_config(page_title="PlanBairros", page_icon="🏙️", layout="wide")
        st.error(f"Dependência ausente: **{e}**.")
        st.stop()

gpd, pdk, DeckFunction, wkb, wkt, shapely_base, pq = _import_stack()

# ====================== config / tema ======================

st.set_page_config(
    page_title="PlanBairros",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PB_NAVY = "#14407D"
ORANGE_RED_GRAD = ["#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#e34a33", "#b30000"]
PLACEHOLDER_VAR = "— selecione uma variável —"
PLACEHOLDER_LIM = "— selecione o limite —"

LOGO_HEIGHT = 135
MAP_HEIGHT = 675
SIMPLIFY_M_SETORES = 25
SIMPLIFY_M_ISOCRONAS = 80
SIMPLIFY_M_OVERLAY = 20

# Overlays permanentes
REF_GREEN = "#2E7D32"
REF_BLUE = "#1E88E5"
REF_DARKGRAY = "#333333"
RIVER_WIDTH_PX = 4.5
RAIL_WIDTH_PX = 6.0

# ===== estilos GL (sem token) =====
CARTO_POSITRON_GL = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

# ===== ESRI Tile URL (raster) =====
ESRI_WORLD_IMAGERY = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}.jpg"
)

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700;900&display=swap');
        html, body, .stApp {{ font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; }}
        .main .block-container {{ padding-top: .15rem !important; padding-bottom: .6rem !important; }}
        .pb-row {{ display:flex; align-items:center; gap:9px; margin-bottom:0; }}
        .pb-logo {{ height:{LOGO_HEIGHT}px; width:auto; display:block; }}
        .pb-header {{ background:{PB_NAVY}; color:#fff; border-radius:14px; padding:14px 15px; width:100%; }}
        .pb-title {{ font-size:2.85rem; font-weight:900; line-height:1.05; letter-spacing:.2px; }}
        .pb-subtitle {{ font-size:1.425rem; opacity:.95; margin-top:5px; }}
        .pb-card {{ background:#fff; border:1px solid rgba(20,64,125,.10); box-shadow:0 1px 2px rgba(0,0,0,.04); border-radius:14px; padding:9px; }}
        .legend-card {{ margin-top:9px; background:#fff; border:1px solid rgba(20,64,125,.10); border-radius:12px; padding:8px 9px; }}
        .legend-title {{ font-weight:800; margin-bottom:4px; }}
        .legend-row {{ display:flex; align-items:center; gap:6px; margin:3px 0; }}
        .legend-swatch {{ width:14px; height:14px; border-radius:4px; display:inline-block; border:1px solid rgba(0,0,0,.15); }}

        .pb-floating-legend {{
            position: fixed; right: 14px; bottom: 14px; z-index: 9999; background: #fff;
            border:1px solid rgba(20,64,125,.10); border-radius:12px; box-shadow:0 2px 6px rgba(0,0,0,.15);
            padding:8px 9px; max-width: 210px; font-size: 10px;
        }}
        .pb-card .stSelectbox > div[data-baseweb="select"],
        .pb-card .stSelectbox div[role="combobox"] {{ min-width: 520px; }}
        [data-baseweb="popover"] {{ width: 520px; max-width: 520px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ====================== caminhos e util ======================
try:
    REPO_ROOT = Path(__file__).resolve().parent
except NameError:
    REPO_ROOT = Path.cwd()

DATA_DIR = (
    REPO_ROOT / "limites_administrativos"
    if (REPO_ROOT / "limites_administrativos").exists()
    else REPO_ROOT
)
GEOM_FILE = DATA_DIR / "IDCenso2023.parquet"
METRICS_FILE = DATA_DIR / "SetoresCensitarios2023.parquet"
LOGO_PATH = REPO_ROOT / "assets" / "logo_todos.jpg"

def _logo_data_uri() -> str:
    if LOGO_PATH.exists():
        b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
        return f"data:image/{LOGO_PATH.suffix.lstrip('.').lower()};base64,{b64}"
    return ("https://raw.githubusercontent.com/streamlit/brand/refs/heads/main/logomark/"
            "streamlit-mark-color.png")

def _is_finite(*vals: float) -> bool:
    return all(np.isfinite(v) for v in vals)

def center_from_bounds(gdf, fallback: Tuple[float, float] = (-23.55, -46.63)) -> Tuple[float, float]:
    try:
        minx, miny, maxx, maxy = gdf.total_bounds
        if not _is_finite(minx, miny, maxx, maxy):
            return fallback
        lat = (miny + maxy) / 2
        lon = (minx + maxx) / 2
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return fallback
        return (lat, lon)
    except Exception:
        return fallback

def _hex_to_rgba(h: str, a: int = 255) -> List[int]:
    h = h.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)] + [a]

# ====================== leitores ======================

def _read_gdf_robusto(path: Path, columns: Optional[List[str]] = None) -> Optional["gpd.GeoDataFrame"]:
    if not path.exists():
        return None
    try:
        gdf = gpd.read_parquet(path, columns=columns)
        if not isinstance(gdf, gpd.GeoDataFrame) or "geometry" not in gdf.columns:
            raise ValueError
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        return gdf.to_crs(4326)
    except Exception:
        try:
            table = pq.read_table(str(path), columns=columns)
            pdf = table.to_pandas()
            geom_col = next((c for c in pdf.columns if c.lower() in {"geometry", "geom", "wkb", "wkt"}), None)
            if geom_col is None:
                return None
            vals = pdf[geom_col]
            if vals.dropna().astype(str).str.startswith(("POLY", "MULTI", "LINE", "POINT")).any():
                geo = vals.dropna().apply(wkt.loads)
            else:
                geo = vals.dropna().apply(lambda b: wkb.loads(b, hex=isinstance(b, str)))
            return gpd.GeoDataFrame(pdf.drop(columns=[geom_col]), geometry=geo, crs=4326)
        except Exception:
            return None

@st.cache_data(show_spinner=False, ttl=3600)
def load_setores_geom() -> Tuple[Optional["gpd.GeoDataFrame"], Optional[str]]:
    if not GEOM_FILE.exists():
        return None, None
    gdf = _read_gdf_robusto(GEOM_FILE, ["fid", "geometry"])
    if gdf is None:
        return None, None
    try:
        gdfm = gdf.to_crs(3857)
        gdfm["geometry"] = gdfm.geometry.simplify(SIMPLIFY_M_SETORES, preserve_topology=True)
        gdf = gdfm.to_crs(4326)
    except Exception:
        pass
    return gdf, "fid"

@st.cache_data(show_spinner=False, ttl=3600, max_entries=64)
def load_metric_column(var_label: str) -> Optional[pd.DataFrame]:
    if not METRICS_FILE.exists():
        return None
    cols = pq.ParquetFile(str(METRICS_FILE)).schema.names
    mapping: Dict[str, List[str] | str] = {
        "População (Pessoa/ha)": ["populacao", "pop", "pessoa_ha"],
        "Densidade demográfica (hab/ha)": ["densidade_demografica", "densidade", "hab_ha"],
        "Elevação média": ["elevacao_media", "elevacao", "elev_med", "altitude_media"],
        "Cluster (perfil urbano)": ["cluster", "classe", "label"],
    }
    cand = mapping.get(var_label, var_label)
    wanted = None
    for c in (cand if isinstance(cand, (list, tuple)) else [cand]):
        for col in cols:
            if re.sub(r"[^a-z0-9]", "", col.lower()) == re.sub(r"[^a-z0-9]", "", str(c).lower()):
                wanted = col
                break
        if wanted:
            break
    if not wanted or "fid" not in cols:
        return None
    table = pq.read_table(str(METRICS_FILE), columns=["fid", wanted])
    df = table.to_pandas().rename(columns={wanted: "__value__"})
    return df

@st.cache_data(show_spinner=False, ttl=3600)
def load_isocronas_raw() -> Optional["gpd.GeoDataFrame"]:
    for name in ("isocronas.parquet", "isócronas.parquet", "Isocronas.parquet"):
        p = DATA_DIR / name
        if p.exists():
            return _read_gdf_robusto(p, None)
    return None

@st.cache_data(show_spinner=False, ttl=3600)
def load_admin_layer(name: str) -> Optional["gpd.GeoDataFrame"]:
    stems = {
        "Distritos": "Distritos.parquet",
        "ZonasOD2023": "ZonasOD2023.parquet",
        "Subprefeitura": "Subprefeitura.parquet",
        "Isócronas": "isocronas.parquet",
    }
    p = DATA_DIR / stems.get(name, "")
    if not p.exists():
        return None
    return _read_gdf_robusto(p, ["geometry"])  # type: ignore[return-value]

# ---------- limpeza de geometrias ----------

def _valid_geoms(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty:
        return gdf
    g = gdf.copy()
    try:
        mask = g.geometry.notna()
        try:
            mask &= ~g.geometry.is_empty
        except Exception:
            pass
        # bounds dentro de WGS84
        def _ok(b):
            if b is None or not _is_finite(*b):
                return False
            minx, miny, maxx, maxy = b
            return (-180 <= minx <= 180 and -180 <= maxx <= 180 and
                    -90 <= miny <= 90 and -90 <= maxy <= 90)
        bnds = g.geometry.bounds
        if not bnds.empty:
            ok_bounds = bnds.apply(lambda row: _ok((row.minx, row.miny, row.maxx, row.maxy)), axis=1)
            mask &= ok_bounds.values
        g = g[mask]
    except Exception:
        pass
    return g

def _explode_keep_types(gdf: "gpd.GeoDataFrame", allowed: Tuple[str, ...]) -> "gpd.GeoDataFrame":
    try:
        g = gdf.explode(index_parts=False, ignore_index=True)
    except Exception:
        g = gdf
    try:
        return g[g.geom_type.isin(allowed)]
    except Exception:
        return g

# ====================== GeoJSON cache ======================

_HASH_FUNCS = {
    gpd.GeoDataFrame: lambda df: (len(df), tuple(df.columns), tuple(np.round(df.total_bounds, 6).tolist())),
}

@st.cache_data(show_spinner=False, ttl=3600, max_entries=256, hash_funcs=_HASH_FUNCS)
def gdf_to_geojson_str(gdf: "gpd.GeoDataFrame", cols: Tuple[str, ...]) -> str:
    g = _valid_geoms(gdf)
    return g[list(cols)].to_json()

def gdf_to_geojson_obj(gdf: "gpd.GeoDataFrame", cols: List[str] | Tuple[str, ...]) -> dict:
    return json.loads(gdf_to_geojson_str(gdf, tuple(cols)))

# ====================== helpers ======================

def _sample_gradient(colors: List[str], n: int) -> List[str]:
    if n <= 1: return [colors[-1]]
    out: List[str] = []
    for i in range(n):
        t = i / (n - 1)
        pos = t * (len(colors) - 1)
        j = min(int(math.floor(pos)), len(colors) - 2)
        frac = pos - j
        def h2r(x): x = x.lstrip("#"); return [int(x[k:k+2], 16) for k in (0, 2, 4)]
        def r2h(r): return "#{:02x}{:02x}{:02x}".format(*r)
        c1, c2 = h2r(colors[j]), h2r(colors[j + 1])
        mix = [int(c1[k] + frac * (c2[k] - c1[k])) for k in range(3)]
        out.append(r2h(mix))
    return out

def _equal_edges(vmin: float, vmax: float, k: int = 6) -> List[float]:
    step = (vmax - vmin) / k
    edges = [vmin + i * step for i in range(k + 1)]
    edges[-1] = vmax
    return edges

def _quantile_edges(vals: np.ndarray, k: int = 6) -> List[float]:
    qs = np.linspace(0, 1, k + 1)
    edges = list(np.quantile(vals, qs))
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    return edges

def classify_auto6(series: pd.Series) -> Tuple[pd.Series, List[Tuple[int, int]], List[str]]:
    s = pd.to_numeric(series, errors="coerce")
    v = s.dropna()
    if v.empty:
        return pd.Series([-1] * len(series), index=series.index), [], []
    vmin, vmax = float(v.min()), float(v.max())
    if vmin == vmax:
        idx = pd.cut(s, bins=[vmin - 1e-9, vmax + 1e-9], labels=False, include_lowest=True)
        palette = _sample_gradient(ORANGE_RED_GRAD, 6)
        return idx.fillna(-1).astype("Int64"), [(int(round(vmin)), int(round(vmax)))], palette
    edges_eq = _equal_edges(vmin, vmax, 6)
    idx_eq = pd.cut(s, bins=[edges_eq[0] - 1e-9] + edges_eq[1:], labels=False, include_lowest=True)
    counts = idx_eq.value_counts(dropna=True)
    non_empty = (counts > 0).sum()
    max_share = (counts.max() / counts.sum()) if counts.sum() > 0 else 1.0
    use_quantile = (max_share > 0.35) or (non_empty < 4)
    if use_quantile:
        edges = _quantile_edges(v.to_numpy(), 6)
        idx = pd.cut(s, bins=[edges[0] - 1e-9] + edges[1:], labels=False, include_lowest=True)
    else:
        edges = edges_eq
        idx = idx_eq
    breaks_int: List[Tuple[int, int]] = []
    for i in range(6):
        a = int(round(edges[i])); b = int(round(edges[i + 1])); b = max(b, a)
        breaks_int.append((a, b))
    palette = _sample_gradient(ORANGE_RED_GRAD, 6)
    return idx.fillna(-1).astype("Int64"), breaks_int, palette

# ====================== UI ======================

def left_controls() -> Dict[str, Any]:
    st.markdown("<div style='margin-top:-6px'></div>", unsafe_allow_html=True)
    st.markdown("### Variáveis (Setores Censitários e Isócronas)")
    var = st.selectbox(
        "Selecione a variável",
        [PLACEHOLDER_VAR, "População (Pessoa/ha)", "Densidade demográfica (hab/ha)",
         "Elevação média", "Cluster (perfil urbano)", "Área de influência de bairro"],
        index=0, key="pb_var", placeholder="Escolha…",
    )

    st.markdown("### Configurações")
    limite = st.selectbox(
        "Limites Administrativos",
        [PLACEHOLDER_LIM, "Distritos", "ZonasOD2023", "Subprefeitura",
         "Isócronas", "Setores Censitários (linhas)"],
        index=0, key="pb_limite", placeholder="Escolha…",
    )

    fundo = st.radio(
        "Fundo do mapa", ["Claro (CARTO)", "Satélite (ESRI)"],
        index=0, horizontal=True, key="pb_basemap",
    )

    st.checkbox("Rótulos permanentes (dinâmicos por zoom)", value=False, key="pb_labels_on")

    st.caption("Use o botão abaixo para limpar os caches de dados em memória.")
    if st.button("🧹 Limpar cache de dados", type="secondary"):
        st.cache_data.clear()
        st.success("Cache limpo. Selecione novamente a camada/variável.")

    sel_now = {"var": var, "lim": limite}
    sel_prev = st.session_state.get("_pb_prev_sel")
    if sel_prev and (sel_prev != sel_now):
        st.cache_data.clear()
    st.session_state["_pb_prev_sel"] = sel_now

    legend_ph = st.empty()
    st.session_state["_legend_ph"] = legend_ph
    return {"variavel": var, "limite": limite, "fundo": fundo}

# ====================== Legendas ======================

def show_numeric_legend(title: str, breaks: List[Tuple[int, int]], palette: List[str]):
    ph = st.session_state.get("_legend_ph")
    if not ph: return
    if not breaks: ph.empty(); return
    rows = []
    for (a, b), col in zip(breaks, palette):
        label = f"{a:,} – {b:,}".replace(",", ".")
        rows.append(f"<div class='legend-row'><span class='legend-swatch' style='background:{col}'></span><span>{label}</span></div>")
    html = f"<div class='legend-card'><div class='legend-title'>{title}</div>{''.join(rows)}</div>"
    ph.markdown(html, unsafe_allow_html=True)

def show_categorical_legend(title: str, items: List[Tuple[str, str]]):
    ph = st.session_state.get("_legend_ph")
    if not ph: return
    rows = [f"<div class='legend-row'><span class='legend-swatch' style='background:{c}'></span><span>{l}</span></div>" for l, c in items]
    html = f"<div class='legend-card'><div class='legend-title'>{title}</div>{''.join(rows)}</div>"
    ph.markdown(html, unsafe_allow_html=True)

def render_reference_legend_floating():
    html = """
    <div class="pb-floating-legend">
      <div class="legend-title">Camadas de referência</div>
      <div class="legend-row"><span class="legend-swatch" style="background:#2E7D32"></span><span>Áreas verdes</span></div>
      <div class="legend-row"><span class="legend-swatch" style="background:#1E88E5"></span><span>Rios</span></div>
      <div class="legend-row"><span class="legend-swatch" style="background:#333333"></span><span>Linhas de trem</span></div>
      <div class="legend-row"><span class="legend-swatch" style="background:#333333"></span><span>Linhas de metrô</span></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def clear_legend():
    ph = st.session_state.get("_legend_ph")
    if ph: ph.empty()

# ====================== Render (pydeck) ======================

_MAP_PLACEHOLDER_KEY = "_pb_map_ph"

def _get_map_placeholder():
    ph = st.session_state.get(_MAP_PLACEHOLDER_KEY)
    if ph is None:
        ph = st.empty()
        st.session_state[_MAP_PLACEHOLDER_KEY] = ph
    return ph

def _esri_tilelayer() -> "pdk.Layer":
    return pdk.Layer(
        "TileLayer",
        id="basemap-esri-sat",
        data=ESRI_WORLD_IMAGERY,
        min_zoom=0, max_zoom=19, tile_size=256, opacity=1.0,
        loadOptions={"image": {"crossOrigin": "anonymous"}, "fetch": {"referrerPolicy": "no-referrer"}},
    )

def collect_reference_overlays() -> List["pdk.Layer"]:
    layers: List[pdk.Layer] = []

    g = load_green_areas()
    if g is not None and not g.empty:
        g = _valid_geoms(g)
        g = _explode_keep_types(g, ("Polygon", "MultiPolygon"))
        if not g.empty:
            gj = gdf_to_geojson_obj(g, ("geometry",))
            layers.append(pdk.Layer("GeoJsonLayer", id="areas-verdes", data=gj,
                                    filled=True, stroked=False, pickable=False,
                                    get_fill_color=_hex_to_rgba(REF_GREEN, 255), get_line_width=0))

    r = load_rivers()
    if r is not None and not r.empty:
        r = _valid_geoms(r)
        r = _explode_keep_types(r, ("LineString", "MultiLineString"))
        if not r.empty:
            gj = gdf_to_geojson_obj(r, ("geometry",))
            layers.append(pdk.Layer("GeoJsonLayer", id="rios", data=gj,
                                    filled=False, stroked=True, pickable=False,
                                    get_line_color=_hex_to_rgba(REF_BLUE, 255),
                                    get_line_width=RIVER_WIDTH_PX,
                                    lineWidthUnits="pixels", lineJointRounded=True, lineCapRounded=True))

    t = load_train_lines()
    if t is not None and not t.empty:
        t = _valid_geoms(t)
        t = _explode_keep_types(t, ("LineString", "MultiLineString"))
        if not t.empty:
            gj = gdf_to_geojson_obj(t, ("geometry",))
            layers.append(pdk.Layer("GeoJsonLayer", id="linhas-trem", data=gj,
                                    filled=False, stroked=True, pickable=False,
                                    get_line_color=_hex_to_rgba(REF_DARKGRAY, 255),
                                    get_line_width=RAIL_WIDTH_PX, lineWidthUnits="pixels",
                                    lineJointRounded=True, lineCapRounded=True))

    m = load_metro_lines()
    if m is not None and not m.empty:
        m = _valid_geoms(m)
        m = _explode_keep_types(m, ("LineString", "MultiLineString"))
        if not m.empty:
            gj = gdf_to_geojson_obj(m, ("geometry",))
            layers.append(pdk.Layer("GeoJsonLayer", id="linhas-metro", data=gj,
                                    filled=False, stroked=True, pickable=False,
                                    get_line_color=_hex_to_rgba(REF_DARKGRAY, 255),
                                    get_line_width=RAIL_WIDTH_PX, lineWidthUnits="pixels",
                                    lineJointRounded=True, lineCapRounded=True))

    return layers

def render_pydeck(
    center: Tuple[float, float],
    gdf_layer: Optional["gpd.GeoDataFrame"],
    limite_gdf: Optional["gpd.GeoDataFrame"],
    *,
    tooltip_field: Optional[str],
    categorical_legend: Optional[List[Tuple[str, str]]] = None,
    numeric_legend: Optional[Tuple[str, List[Tuple[int, int]], List[str]]] = None,
    draw_setores_outline: bool = False,
    basemap: str = "Claro (CARTO)",
) -> None:
    layers: List[pdk.Layer] = []

    # Basemap: CARTO em map_style; se ESRI, coloca raster por cima.
    base_style = CARTO_POSITRON_GL
    if basemap == "Satélite (ESRI)":
        layers.append(_esri_tilelayer())

    # Camada temática
    if gdf_layer is not None and not gdf_layer.empty and "__rgba__" in gdf_layer.columns:
        gdf_layer = _valid_geoms(gdf_layer)
        gdf_layer = _explode_keep_types(gdf_layer, ("Polygon", "MultiPolygon", "LineString", "MultiLineString", "Point", "MultiPoint"))
        if not gdf_layer.empty:
            cols = ["geometry", "__rgba__"] + ([tooltip_field] if tooltip_field else [])
            gj = gdf_to_geojson_obj(gdf_layer, tuple(cols))
            layers.append(pdk.Layer("GeoJsonLayer", id="tematica", data=gj,
                                    filled=True, stroked=False, pickable=bool(tooltip_field),
                                    auto_highlight=True, get_fill_color="properties.__rgba__", get_line_width=0))

    # Limite/contorno
    if limite_gdf is not None and not limite_gdf.empty:
        limite_gdf = _valid_geoms(limite_gdf)
        limite_gdf = _explode_keep_types(limite_gdf, ("Polygon", "MultiPolygon", "LineString", "MultiLineString"))
        if not limite_gdf.empty:
            gj_lim = gdf_to_geojson_obj(limite_gdf, ("geometry",))
            layers.append(pdk.Layer("GeoJsonLayer", id="limite", data=gj_lim,
                                    filled=False, stroked=True,
                                    get_line_color=[20, 20, 20, 180],
                                    get_line_width=1, lineWidthUnits="pixels"))

    if draw_setores_outline:
        try:
            geoms_only, _ = load_setores_geom()
        except Exception:
            geoms_only = None
        if geoms_only is not None and not geoms_only.empty:
            geoms_only = _valid_geoms(geoms_only)
            geoms_only = _explode_keep_types(geoms_only, ("Polygon", "MultiPolygon", "LineString", "MultiLineString"))
            if not geoms_only.empty:
                gj = gdf_to_geojson_obj(geoms_only, ("geometry",))
                layers.append(pdk.Layer("GeoJsonLayer", id="setores-outline", data=gj,
                                        filled=False, stroked=True,
                                        get_line_color=[80, 80, 80, 160],
                                        get_line_width=0.6, lineWidthUnits="pixels"))

    # OVERLAYS
    layers.extend(collect_reference_overlays())

    # Deck (sem map_provider; map_style string → MapLibre; sem token)
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=center[0], longitude=center[1], zoom=11, bearing=0, pitch=0),
        map_style=base_style,
        tooltip={"text": f"{{{tooltip_field}}}"} if tooltip_field else None,
    )

    ph = _get_map_placeholder()
    ph.empty()
    ph.pydeck_chart(deck, use_container_width=True)

    # Legendas
    if categorical_legend is not None:
        show_categorical_legend("Área de influência de bairro", categorical_legend)
    elif numeric_legend is not None:
        title, breaks, palette = numeric_legend
        show_numeric_legend(title, breaks, palette)
    else:
        clear_legend()

    render_reference_legend_floating()

# ====================== App ======================

def main() -> None:
    inject_css()
    st.markdown(
        f"""
        <div class="pb-row">
            <img class="pb-logo" src="{_logo_data_uri()}" alt="PlanBairros logo"/>
            <div class="pb-header">
                <div class="pb-title">PlanBairros</div>
                <div class="pb-subtitle">Plataforma de visualização e planejamento em escala de bairro</div>
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

    with map_col:
        center = (-23.55, -46.63)
        var = ui["variavel"]

        # Limite administrativo
        limite_gdf = None
        draw_setores_outline = ui["limite"] == "Setores Censitários (linhas)"
        if ui["limite"] not in (PLACEHOLDER_LIM, "Setores Censitários (linhas)"):
            limite_gdf = load_admin_layer(ui["limite"])
            if limite_gdf is not None and len(limite_gdf) > 0:
                center = center_from_bounds(limite_gdf)

        # ====== VARIÁVEIS ======
        if var == "Área de influência de bairro":
            lut_color = {0:"#542788",1:"#f7f7f7",2:"#d8daeb",3:"#b35806",4:"#b2abd2",5:"#8073ac",6:"#fdb863",7:"#7f3b08",8:"#e08214",9:"#fee0b6"}
            lut_label = {
                0: "Predominância uso misto", 1: "Zona de transição local", 2: "Periférico residencial de média densidade",
                3: "Transição central verticalizada", 4: "Periférico adensado em transição", 5: "Centralidade comercial e de serviços",
                6: "Predominância residencial média densidade", 7: "Áreas íngremes e de encosta", 8: "Alta densidade residencial", 9: "Central verticalizado",
            }
            iso_raw = load_isocronas_raw()
            gdf_layer = None
            legend_items: Optional[List[Tuple[str, str]]] = None
            if iso_raw is not None and not iso_raw.empty:
                cls = next((c for c in iso_raw.columns if re.sub(r"[^a-z0-9]","",c.lower()) in
                            {"isocrona","novaclass","classe","class","cat","category"}), None)
                if cls:
                    g = iso_raw[[cls, "geometry"]].copy()
                    k = pd.to_numeric(g[cls], errors="coerce").astype("Int64")
                    g = g[~k.isna()].copy(); g["__k__"] = k
                    try:
                        gm = g.to_crs(3857)
                        gm["geometry"] = gm.buffer(0).geometry.simplify(SIMPLIFY_M_ISOCRONAS, preserve_topology=True)
                        g = gm.to_crs(4326).explode(index_parts=False, ignore_index=True)
                    except Exception:
                        pass
                    g = _valid_geoms(g)
                    g = _explode_keep_types(g, ("Polygon","MultiPolygon"))
                    if not g.empty:
                        g["__rgba__"] = g["__k__"].map(lambda v: _hex_to_rgba(lut_color.get(int(v), "#c8c8c8"), 200))
                        g["__label__"] = g["__k__"].map(lambda v: f"{int(v)} – {lut_label.get(int(v), 'Outros')}")
                        gdf_layer = g[["geometry", "__rgba__", "__label__"]]
                        center = center_from_bounds(g)
                        present = sorted({int(v) for v in g["__k__"].dropna().unique().tolist()})
                        legend_items = [(f"{k} – {lut_label[k]}", lut_color[k]) for k in present]

            render_pydeck(center, gdf_layer, limite_gdf,
                          tooltip_field="__label__", categorical_legend=legend_items,
                          numeric_legend=None, draw_setores_outline=draw_setores_outline,
                          basemap=ui["fundo"])
            return

        # Demais variáveis (setores)
        gdf_layer = None
        if var != PLACEHOLDER_VAR:
            geoms, id_col = load_setores_geom()
            if geoms is not None and id_col == "fid":
                metric = load_metric_column(var)
                if metric is not None:
                    joined = geoms.merge(metric, on="fid", how="left")
                    center = center_from_bounds(joined)

                    if var == "Cluster (perfil urbano)":
                        cmap = {0:"#bf7db2",1:"#f7bd6a",2:"#cf651f",3:"#ede4e6",4:"#793393"}
                        labels = {
                            0: "1 - Periférico com predominância residencial de alta densidade construtiva",
                            1: "2 - Uso misto de média densidade construtiva",
                            2: "3 - Periférico com predominância residencial de média densidade construtiva",
                            3: "4 - Verticalizado de uso-misto",
                            4: "5 - Predominância de uso comercial e serviços",
                        }
                        s = pd.to_numeric(joined["__value__"], errors="coerce")
                        joined["__rgba__"] = s.map(lambda v: _hex_to_rgba(cmap.get(int(v) if pd.notna(v) else -1, "#c8c8c8"), 200))
                        joined["__label__"] = s.map(lambda v: labels.get(int(v), "Outros") if pd.notna(v) else "Sem dado")
                        g = joined[["geometry", "__rgba__", "__label__"]]
                        g = _valid_geoms(g); g = _explode_keep_types(g, ("Polygon","MultiPolygon"))
                        if not g.empty:
                            gdf_layer = g
                            render_pydeck(center, gdf_layer, limite_gdf,
                                          tooltip_field="__label__", categorical_legend=[(labels[k], cmap[k]) for k in sorted(cmap)],
                                          numeric_legend=None, draw_setores_outline=draw_setores_outline,
                                          basemap=ui["fundo"])
                            return
                    else:
                        classes, breaks_int, palette = classify_auto6(joined["__value__"])
                        color_map = {i: _hex_to_rgba(palette[i], 200) for i in range(6)}
                        joined["__rgba__"] = classes.map(lambda k: color_map.get(int(k) if pd.notna(k) else -1, _hex_to_rgba("#c8c8c8", 200)))
                        joined["__label__"] = pd.to_numeric(joined["__value__"], errors="coerce").round(0).astype("Int64").astype(str)
                        g = joined[["geometry", "__rgba__", "__label__"]]
                        g = _valid_geoms(g); g = _explode_keep_types(g, ("Polygon","MultiPolygon"))
                        if not g.empty:
                            gdf_layer = g
                            render_pydeck(center, gdf_layer, limite_gdf,
                                          tooltip_field="__label__", categorical_legend=None,
                                          numeric_legend=(var, breaks_int, palette),
                                          draw_setores_outline=draw_setores_outline,
                                          basemap=ui["fundo"])
                            return

        # sem variável: só fundo + overlays
        render_pydeck(center, None, limite_gdf,
                      tooltip_field=None, categorical_legend=None, numeric_legend=None,
                      draw_setores_outline=draw_setores_outline, basemap=ui["fundo"])

if __name__ == "__main__":
    main()
