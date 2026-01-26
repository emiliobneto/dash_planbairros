# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Imports geoespaciais (lazy-safe)
# =========================
def _import_geo():
    import geopandas as gpd
    import pydeck as pdk
    from shapely.geometry import mapping  # noqa: F401
    import shapely
    return gpd, pdk, shapely

gpd, pdk, shapely = _import_geo()

# =========================
# Config UI (estética "anterior")
# =========================
st.set_page_config(page_title="PlanBairros — Dash", page_icon="🗺️", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.0rem; padding-bottom: 1.0rem; }
      section[data-testid="stSidebar"] { width: 340px !important; }
      .st-emotion-cache-1y4p8pa { padding: 1.0rem; } /* sidebar padding */
      .pb-note { font-size: 0.90rem; opacity: 0.85; }
      .pb-badge { display:inline-block; padding:2px 8px; border-radius:999px; background:#1118270d; margin-right:6px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Paths (respeitando GitHub)
# =========================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "limites_administrativos"  # dash_planbairros/limites_administrativos/

FILES: Dict[str, Path] = {
    # limites (parquet)
    "Distritos": DATA_DIR / "Distritos.parquet",
    "Subprefeitura": DATA_DIR / "subprefeitura.parquet",
    "Zonas OD 2023": DATA_DIR / "ZonasOD2023.parquet",
    "Habitação Precária": DATA_DIR / "HabitacaoPrecaria.parquet",
    "Isócronas": DATA_DIR / "isocronas.parquet",
    # base p/ vínculo fid
    "ID Censo 2023": DATA_DIR / "IDCenso2023.parquet",
    "Setores Censitários 2023": DATA_DIR / "SetoresCensitarios2023.parquet",
    # overlays (geojson)
    "Áreas verdes": DATA_DIR / "area_verde.geojson",
    "Rios": DATA_DIR / "rios.geojson",
    "Linhas Metrô": DATA_DIR / "linhas_metro.geojson",
    "Linhas Trem": DATA_DIR / "linhas_trem.geojson",
}

# =========================
# Basemaps
# =========================
CARTO_POSITRON_GL = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
# (Google pode falhar por CORS/termos; deixo como opção, mas recomendo ESRI se quiser satélite estável)
GOOGLE_SAT_TILES = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
ESRI_WORLD_IMAGERY = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

# =========================
# Helpers CRS / geometria
# =========================
def _bounds_look_projected(bounds: Tuple[float, float, float, float]) -> bool:
    minx, miny, maxx, maxy = bounds
    # Se passar muito de graus, provavelmente está em metros (UTM etc.)
    return (abs(minx) > 180) or (abs(maxx) > 180) or (abs(miny) > 90) or (abs(maxy) > 90)

def _ensure_crs(gdf: "gpd.GeoDataFrame", assume_epsg_if_missing: int = 31983) -> "gpd.GeoDataFrame":
    """Garante CRS e converte para EPSG:4326 (Carto/MapLibre)."""
    if gdf is None or gdf.empty:
        return gdf

    # Se CRS ausente, tenta inferir por bounds
    if gdf.crs is None:
        try:
            b = gdf.total_bounds
            if _bounds_look_projected(tuple(b)):
                gdf = gdf.set_crs(epsg=assume_epsg_if_missing, allow_override=True)
            else:
                gdf = gdf.set_crs(epsg=4326, allow_override=True)
        except Exception:
            # fallback seguro
            gdf = gdf.set_crs(epsg=assume_epsg_if_missing, allow_override=True)

    # Converte para 4326
    try:
        if str(gdf.crs).lower() not in ("epsg:4326", "wgs84"):
            gdf = gdf.to_crs(epsg=4326)
    except Exception:
        # Se conversão falhar, tenta forçar 4326 (último recurso)
        gdf = gdf.set_crs(epsg=4326, allow_override=True)

    return gdf

def _clean_geoms(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty:
        return gdf
    gdf = gdf[gdf.geometry.notna()].copy()
    # make_valid (Shapely 2)
    try:
        gdf["geometry"] = shapely.make_valid(gdf.geometry)
    except Exception:
        pass
    # remove geometrias vazias
    try:
        gdf = gdf[~gdf.geometry.is_empty].copy()
    except Exception:
        pass
    return gdf

def _maybe_simplify(gdf: "gpd.GeoDataFrame", max_features: int = 20000, tol_m: float = 15.0) -> "gpd.GeoDataFrame":
    """
    Simplificação leve para evitar travar render quando há muitas feições.
    Faz em EPSG:3857 (metros) e volta para 4326.
    """
    if gdf is None or gdf.empty:
        return gdf
    if len(gdf) <= max_features:
        return gdf

    # amostra para não explodir memória
    gdf = gdf.sample(max_features, random_state=42)

    try:
        g3857 = gdf.to_crs(epsg=3857)
        g3857["geometry"] = g3857.geometry.simplify(tol_m, preserve_topology=True)
        gdf = g3857.to_crs(epsg=4326)
    except Exception:
        # se falhar, segue sem simplificar
        pass
    return gdf

# =========================
# Loaders (cache)
# =========================
@st.cache_data(show_spinner=False)
def load_parquet(path_str: str) -> "gpd.GeoDataFrame":
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path.as_posix()}")
    gdf = gpd.read_parquet(path)
    gdf = _clean_geoms(gdf)
    gdf = _ensure_crs(gdf)
    return gdf

@st.cache_data(show_spinner=False)
def load_geojson(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path.as_posix()}")
    gdf = gpd.read_file(path)
    gdf = _clean_geoms(gdf)
    gdf = _ensure_crs(gdf)
    return gdf.__geo_interface__

def load_green_areas() -> Optional[Dict[str, Any]]:
    try:
        return load_geojson(str(FILES["Áreas verdes"]))
    except Exception:
        return None

def load_rivers() -> Optional[Dict[str, Any]]:
    try:
        return load_geojson(str(FILES["Rios"]))
    except Exception:
        return None

def load_metro() -> Optional[Dict[str, Any]]:
    try:
        return load_geojson(str(FILES["Linhas Metrô"]))
    except Exception:
        return None

def load_train() -> Optional[Dict[str, Any]]:
    try:
        return load_geojson(str(FILES["Linhas Trem"]))
    except Exception:
        return None

# =========================
# Jenks (Natural Breaks) — sem libs extras
# =========================
def jenks_breaks(values: np.ndarray, n_classes: int = 6) -> Optional[List[float]]:
    """
    Jenks natural breaks (DP). Para performance, use em amostra.
    Retorna lista de breaks com tamanho n_classes+1.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < n_classes + 1:
        return None

    v.sort()
    n_data = v.size

    # Matrizes
    lower = np.zeros((n_data + 1, n_classes + 1), dtype=int)
    var = np.full((n_data + 1, n_classes + 1), np.inf, dtype=float)

    for j in range(1, n_classes + 1):
        lower[0, j] = 1
        var[0, j] = 0.0

    for i in range(1, n_data + 1):
        lower[i, 1] = 1

    # DP
    for l in range(1, n_data + 1):
        s1 = s2 = w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = v[i3 - 1]
            s2 += val * val
            s1 += val
            w += 1.0
            vv = s2 - (s1 * s1) / w
            i4 = i3 - 1

            if i4 != 0:
                for j in range(2, n_classes + 1):
                    if var[l, j] >= vv + var[i4, j - 1]:
                        lower[l, j] = i3
                        var[l, j] = vv + var[i4, j - 1]

        var[l, 1] = vv

    # Backtrack
    k = n_data
    kclass = [0.0] * (n_classes + 1)
    kclass[n_classes] = float(v[-1])
    kclass[0] = float(v[0])

    count = n_classes
    while count >= 2:
        idx = lower[k, count] - 1
        kclass[count - 1] = float(v[idx])
        k = lower[k, count] - 1
        count -= 1

    # Se breaks degenerarem (valores repetidos), retorna None
    if any(kclass[i] > kclass[i + 1] for i in range(len(kclass) - 1)):
        return None
    if len(set(kclass)) < len(kclass):
        return None

    return kclass

# =========================
# Cores
# =========================
PALETTE_6 = [
    [247, 251, 255, 220],
    [222, 235, 247, 220],
    [198, 219, 239, 220],
    [158, 202, 225, 220],
    [107, 174, 214, 220],
    [33, 113, 181, 220],
]

# =========================
# ViewState helper
# =========================
def _zoom_from_bounds(bounds: Tuple[float, float, float, float]) -> float:
    # aproximação simples
    minx, miny, maxx, maxy = bounds
    dx = maxx - minx
    dy = maxy - miny
    span = max(dx, dy)
    if span <= 0:
        return 11.0
    # 360 graus ~ zoom 0; cada zoom dobra resolução
    z = np.log2(360.0 / span)
    return float(np.clip(z, 9.0, 15.5))

def _viewstate_from_gdf(gdf_any: Optional["gpd.GeoDataFrame"]) -> "pdk.ViewState":
    # default SP
    lat, lon, zoom = -23.5505, -46.6333, 10.8
    if gdf_any is not None and (not gdf_any.empty):
        try:
            b = gdf_any.total_bounds
            lon = float((b[0] + b[2]) / 2.0)
            lat = float((b[1] + b[3]) / 2.0)
            zoom = _zoom_from_bounds(tuple(b))
        except Exception:
            pass
    return pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom, pitch=0)

# =========================
# Construção de camadas (somente GeoJsonLayer p/ evitar erro do PathLayer)
# =========================
def layer_admin_outline(gdf: "gpd.GeoDataFrame", layer_id: str) -> "pdk.Layer":
    gj = gdf.__geo_interface__
    return pdk.Layer(
        "GeoJsonLayer",
        data=gj,
        id=layer_id,
        stroked=True,
        filled=False,  # limites: só linha
        get_line_color=[20, 20, 20, 220],
        lineWidthUnits="pixels",
        get_line_width=1.2,
        pickable=True,
        auto_highlight=True,
    )

def layer_variable_fill(gdf: "gpd.GeoDataFrame", layer_id: str) -> "pdk.Layer":
    # gdf precisa ter colunas: fill_color e line_color
    gj = gdf.__geo_interface__
    return pdk.Layer(
        "GeoJsonLayer",
        data=gj,
        id=layer_id,
        stroked=True,
        filled=True,
        get_fill_color="fill_color",
        get_line_color=[40, 40, 40, 90],
        lineWidthUnits="pixels",
        get_line_width=0.6,
        pickable=True,
        auto_highlight=True,
    )

def layer_lines_geojson(gj: Dict[str, Any], layer_id: str, rgba: List[int], width: float = 2.0) -> "pdk.Layer":
    return pdk.Layer(
        "GeoJsonLayer",
        data=gj,
        id=layer_id,
        stroked=True,
        filled=False,
        get_line_color=rgba,
        lineWidthUnits="pixels",
        get_line_width=width,
        pickable=False,
    )

def layer_green_areas(gj: Dict[str, Any], layer_id: str) -> "pdk.Layer":
    # Verde com opacidade 100% (alpha 255), acima de tudo
    return pdk.Layer(
        "GeoJsonLayer",
        data=gj,
        id=layer_id,
        stroked=False,
        filled=True,
        get_fill_color=[0, 160, 70, 255],
        pickable=False,
    )

def layer_tile(tile_url: str, opacity: float, layer_id: str) -> "pdk.Layer":
    return pdk.Layer(
        "TileLayer",
        data=tile_url,
        id=layer_id,
        opacity=opacity,
        tileSize=256,
        # renderSubLayers interno (deck.gl) — pydeck resolve automaticamente
    )

# =========================
# Join fid: IDCenso (geom) + Setores (atributos)
# =========================
@st.cache_data(show_spinner=False)
def load_setores_joined() -> "gpd.GeoDataFrame":
    g_id = load_parquet(str(FILES["ID Censo 2023"]))
    g_set = load_parquet(str(FILES["Setores Censitários 2023"]))

    # garante coluna fid (case-insensitive)
    def _find_fid(cols: List[str]) -> Optional[str]:
        for c in cols:
            if c.lower() == "fid":
                return c
        return None

    fid_id = _find_fid(list(g_id.columns))
    fid_set = _find_fid(list(g_set.columns))

    if fid_id is None:
        # tenta usar índice
        g_id = g_id.reset_index().rename(columns={"index": "fid"})
        fid_id = "fid"
    if fid_set is None:
        g_set = g_set.reset_index().rename(columns={"index": "fid"})
        fid_set = "fid"

    # normaliza tipo
    g_id[fid_id] = pd.to_numeric(g_id[fid_id], errors="coerce").astype("Int64")
    g_set[fid_set] = pd.to_numeric(g_set[fid_set], errors="coerce").astype("Int64")

    # atributos: remove geometry duplicada do Setores (mantém a do IDCenso)
    attr = g_set.drop(columns=[c for c in g_set.columns if c.lower() == "geometry"], errors="ignore").copy()

    # dedup
    attr = attr.drop_duplicates(subset=[fid_set])
    g_id = g_id.drop_duplicates(subset=[fid_id])

    merged = g_id.merge(attr, left_on=fid_id, right_on=fid_set, how="left", suffixes=("", "_set"))

    merged = _clean_geoms(merged)
    merged = _ensure_crs(merged)
    return merged

def numeric_variables(df: "gpd.GeoDataFrame") -> List[str]:
    banned = {"geometry", "fid", "FID"}
    cols = []
    for c in df.columns:
        if c in banned:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def classify_jenks_6(gdf_in: "gpd.GeoDataFrame", var: str) -> Tuple["gpd.GeoDataFrame", Optional[List[float]]]:
    gdf = gdf_in.copy()

    s = pd.to_numeric(gdf[var], errors="coerce")
    vals = s.to_numpy(dtype=float)

    # amostra p/ Jenks (performance)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        gdf["fill_color"] = [[0, 0, 0, 0]] * len(gdf)
        return gdf, None

    sample = finite
    if sample.size > 20000:
        rng = np.random.default_rng(42)
        sample = rng.choice(sample, size=20000, replace=False)

    br = jenks_breaks(sample, 6)
    if br is None:
        # fallback quantis (não desejado, mas evita quebrar)
        q = np.nanquantile(finite, [0, 1/6, 2/6, 3/6, 4/6, 5/6, 1.0])
        br = [float(x) for x in q]

    # bins: 6 classes
    bins = np.array(br[1:-1], dtype=float)
    cls = np.digitize(vals, bins, right=True)  # 0..5
    cls = np.where(np.isfinite(vals), cls, -1)

    colors = []
    for c in cls:
        if c < 0:
            colors.append([0, 0, 0, 0])
        else:
            colors.append(PALETTE_6[int(c)])

    gdf["fill_color"] = colors
    return gdf, br

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.markdown("## 🗺️ PlanBairros — Mapa")
    st.markdown('<div class="pb-note">Camadas e variáveis a partir de <b>dash_planbairros/limites_administrativos/</b>.</div>', unsafe_allow_html=True)

    if st.button("🧹 Limpar cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache limpo. Verifique novamente as camadas.")

    st.markdown("### Fundo do mapa")
    basemap = st.selectbox(
        "Selecione o fundo",
        ["Carto Positron", "ESRI World Imagery (satélite)", "Google Satellite (50% opacidade)"],
        index=0,
    )

    st.markdown("### Limites administrativos")
    draw_admin = st.toggle("Exibir limites", value=True)
    admin_choice = st.selectbox(
        "Camada de limite",
        ["Distritos", "Subprefeitura", "Zonas OD 2023", "Habitação Precária", "Isócronas", "ID Censo 2023", "Setores Censitários 2023"],
        index=0,
        disabled=not draw_admin,
    )

    st.markdown("### Variáveis (Setores Censitários)")
    draw_var = st.toggle("Exibir variável (choropleth)", value=False)

    var_name = None
    if draw_var:
        try:
            setores_join = load_setores_joined()
            num_vars = numeric_variables(setores_join)
            if len(num_vars) == 0:
                st.warning("Nenhuma variável numérica mostrou-se disponível em SetoresCensitarios2023.")
            else:
                var_name = st.selectbox("Variável", num_vars, index=0)
        except Exception as e:
            st.error(f"Falha ao carregar vínculo fid (IDCenso ↔ Setores): {e}")
            draw_var = False

    st.markdown("### Referências (overlays)")
    show_rios = st.checkbox("Rios (azul)", value=True)
    show_metro = st.checkbox("Linhas Metrô (preto)", value=True)
    show_trem = st.checkbox("Linhas Trem (preto)", value=True)
    show_verde = st.checkbox("Áreas verdes (verde, opacidade 100%)", value=True)

# =========================
# Preparar camadas
# =========================
layers: List["pdk.Layer"] = []

# Basemap
map_style = None
if basemap == "Carto Positron":
    map_style = CARTO_POSITRON_GL
elif basemap == "ESRI World Imagery (satélite)":
    layers.append(layer_tile(ESRI_WORLD_IMAGERY, opacity=1.0, layer_id="tile-esri"))
    map_style = None
else:
    # Google 50% opacidade
    layers.append(layer_tile(GOOGLE_SAT_TILES, opacity=0.5, layer_id="tile-google"))
    map_style = None

# Admin outline (só linha)
admin_gdf_for_view = None
if draw_admin:
    try:
        g_admin = load_parquet(str(FILES[admin_choice]))
        g_admin = _maybe_simplify(g_admin, max_features=20000, tol_m=15.0)
        admin_gdf_for_view = g_admin
        layers.append(layer_admin_outline(g_admin, layer_id="admin-outline"))
    except Exception as e:
        st.error(f"Não foi possível carregar '{admin_choice}': {e}")

# Variable (choropleth) — usa IDCenso (geom) + Setores (attrs) via fid e Jenks 6 classes
var_gdf_for_view = None
legend_breaks = None
if draw_var and var_name:
    try:
        setores_join = load_setores_joined()
        setores_join = _maybe_simplify(setores_join, max_features=20000, tol_m=10.0)
        g_var, legend_breaks = classify_jenks_6(setores_join, var_name)
        var_gdf_for_view = g_var
        layers.append(layer_variable_fill(g_var, layer_id="var-fill"))
    except Exception as e:
        st.error(f"Erro ao renderizar variável '{var_name}': {e}")

# Overlays (linhas)
if show_rios:
    gj = load_rivers()
    if gj:
        layers.append(layer_lines_geojson(gj, "rios", rgba=[0, 120, 255, 255], width=2.2))

if show_metro:
    gj = load_metro()
    if gj:
        layers.append(layer_lines_geojson(gj, "metro", rgba=[0, 0, 0, 255], width=2.4))

if show_trem:
    gj = load_train()
    if gj:
        layers.append(layer_lines_geojson(gj, "trem", rgba=[0, 0, 0, 255], width=2.4))

# Áreas verdes por cima de tudo
if show_verde:
    gj = load_green_areas()
    if gj:
        layers.append(layer_green_areas(gj, "verde"))

# =========================
# ViewState
# =========================
view_ref = var_gdf_for_view if (var_gdf_for_view is not None) else admin_gdf_for_view
view_state = _viewstate_from_gdf(view_ref)

# Tooltip (mantém simples e estável)
tooltip = {"text": "{name}"}
if draw_var and var_name:
    tooltip = {"text": f"{var_name}: " + "{" + var_name + "}"}

deck = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    map_style=map_style,
    tooltip=tooltip,
)

# =========================
# Layout principal
# =========================
col_map, col_info = st.columns([0.72, 0.28], gap="large")

with col_map:
    st.pydeck_chart(deck, use_container_width=True)

with col_info:
    st.markdown("### Estado das camadas")
    st.markdown(f'<span class="pb-badge">Fundo</span> {basemap}', unsafe_allow_html=True)
    st.markdown(f'<span class="pb-badge">Limites</span> {"ON" if draw_admin else "OFF"}', unsafe_allow_html=True)
    if draw_admin:
        st.write(f"• Camada: **{admin_choice}**")
    st.markdown(f'<span class="pb-badge">Variável</span> {"ON" if draw_var else "OFF"}', unsafe_allow_html=True)
    if draw_var and var_name:
        st.write(f"• Campo: **{var_name}**")
        if legend_breaks:
            st.markdown("**Quebras (Jenks, 6 classes)**")
            # Exibe intervalos
            for i in range(6):
                a = legend_breaks[i]
                b = legend_breaks[i + 1]
                st.write(f"• Classe {i+1}: {a:.4g} → {b:.4g}")
        else:
            st.info("Quebras indisponíveis (variável vazia ou fallback).")

    st.markdown("### Referências")
    st.write(f"• Rios: {'ON' if show_rios else 'OFF'}")
    st.write(f"• Metrô: {'ON' if show_metro else 'OFF'}")
    st.write(f"• Trem: {'ON' if show_trem else 'OFF'}")
    st.write(f"• Áreas verdes: {'ON' if show_verde else 'OFF'}")

# =========================
# Checagem rápida de paths (diagnóstico)
# =========================
with st.expander("Diagnóstico de caminhos (GitHub) — limites_administrativos"):
    st.write(f"ROOT: `{ROOT.as_posix()}`")
    st.write(f"DATA_DIR: `{DATA_DIR.as_posix()}`")
    missing = [k for k, p in FILES.items() if not p.exists()]
    if missing:
        st.error("Arquivos ausentes:")
        for k in missing:
            st.write(f"• {k}: `{FILES[k].as_posix()}`")
    else:
        st.success("Todos os arquivos esperados foram encontrados no caminho informado.")
