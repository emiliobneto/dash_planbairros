# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
import logging

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# LOGGING (vai para os logs do Streamlit Cloud)
# ============================================================
logger = logging.getLogger("dash_planbairros")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

# ============================================================
# IMPORTS GIS (falha amigável)
# ============================================================
try:
    import geopandas as gpd
    import pydeck as pdk
    from shapely.geometry.base import BaseGeometry
    from shapely.errors import GEOSException
    try:
        # shapely 2.x
        from shapely import make_valid  # type: ignore
    except Exception:
        make_valid = None  # type: ignore
except Exception as e:
    st.error("Falha ao importar dependências GIS (geopandas/pydeck/shapely).")
    st.exception(e)
    raise

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Dash PlanBairros", layout="wide")

ROOT = Path(__file__).resolve().parent

PATH_LIMITES = ROOT / "limites_administrativos"
PATH_VARIAVEIS = ROOT / "variaveis"  # se você usa outro nome, ajuste aqui
PATH_REFERENCIAS = ROOT / "referencias"  # opcional

MAX_FEATURES_DEFAULT = 15000

# ============================================================
# HELPERS — Basemaps (MapLibre + Raster)
# ============================================================
def _map_style_raster_xyz(tiles_url: str, opacity: float = 1.0, attribution: str = "") -> Dict[str, Any]:
    # MapLibre style spec básico com fonte raster XYZ
    return {
        "version": 8,
        "sources": {
            "raster-xyz": {
                "type": "raster",
                "tiles": [tiles_url],
                "tileSize": 256,
                "attribution": attribution,
            }
        },
        "layers": [
            {
                "id": "raster-xyz",
                "type": "raster",
                "source": "raster-xyz",
                "paint": {"raster-opacity": float(opacity)},
            }
        ],
        # glyphs evita warnings em alguns renders
        "glyphs": "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
    }


BASEMAPS: Dict[str, Any] = {
    "Carto Positron (leve)": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    "Carto DarkMatter": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    "ESRI World Imagery (satélite)": _map_style_raster_xyz(
        "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        opacity=0.50,
        attribution="Tiles © Esri",
    ),
    # Se você insistir em Google, este tile costuma funcionar sem token em alguns ambientes,
    # mas pode ser bloqueado dependendo de políticas/restrições.
    "Google Satellite (pode bloquear)": _map_style_raster_xyz(
        "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        opacity=0.50,
        attribution="© Google",
    ),
}

# ============================================================
# HELPERS — Arquivos / Geometrias
# ============================================================
def _find_parquet(base_dir: Path, candidates: List[str]) -> Optional[Path]:
    """
    Encontra um parquet por lista de candidatos (tolerante a variações de nome).
    Retorna o primeiro existente.
    """
    for name in candidates:
        p = base_dir / name
        if p.exists():
            return p
        # tolerância: tentar variações com/sem .parquet
        if not name.lower().endswith(".parquet"):
            p2 = base_dir / f"{name}.parquet"
            if p2.exists():
                return p2
    return None


def _ensure_crs_wgs84(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    if gdf.crs is None:
        # Se seu dado já está em WGS84, ok. Se não estiver, ajuste aqui.
        # Eu prefiro falhar “suave” sem quebrar, mas logando.
        logger.warning("GeoDataFrame sem CRS. Assumindo EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    if str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def _sanitize_geometries(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """
    Remove geometria nula/vazia e tenta corrigir inválidas (make_valid quando disponível).
    """
    if gdf.empty:
        return gdf

    gdf = gdf[gdf.geometry.notna()].copy()
    if gdf.empty:
        return gdf

    # remove empties
    try:
        gdf = gdf[~gdf.geometry.is_empty].copy()
    except Exception:
        # se algo estranho acontecer, não quebra o app
        pass

    # tenta corrigir inválidas
    try:
        invalid_mask = ~gdf.geometry.is_valid
        if invalid_mask.any():
            logger.info("Corrigindo geometrias inválidas: %s", int(invalid_mask.sum()))
            if make_valid is not None:
                gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].apply(
                    lambda geom: make_valid(geom) if geom is not None else None
                )
            else:
                # fallback leve: buffer(0) (às vezes resolve)
                gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].buffer(0)
    except Exception as e:
        logger.warning("Falha ao validar/corrigir geometrias (seguindo sem quebrar): %s", e)

    # remove novamente nulos/vazios após correção
    gdf = gdf[gdf.geometry.notna()].copy()
    try:
        gdf = gdf[~gdf.geometry.is_empty].copy()
    except Exception:
        pass

    return gdf


def _maybe_simplify(gdf: "gpd.GeoDataFrame", tolerance: float) -> "gpd.GeoDataFrame":
    if tolerance <= 0 or gdf.empty:
        return gdf
    try:
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.simplify(tolerance, preserve_topology=True)
        gdf = _sanitize_geometries(gdf)
        return gdf
    except Exception as e:
        logger.warning("Simplify falhou (seguindo sem quebrar): %s", e)
        return gdf


def _limit_features(gdf: "gpd.GeoDataFrame", max_features: int) -> Tuple["gpd.GeoDataFrame", Optional[str]]:
    """
    Limita número de features para evitar travar o front.
    """
    if gdf.empty:
        return gdf, None
    if max_features <= 0:
        return gdf, None
    if len(gdf) <= max_features:
        return gdf, None

    msg = f"Seleção muito grande ({len(gdf):,} polígonos). Exibindo amostra de {max_features:,}."
    logger.warning(msg)

    # amostra estável
    gdf2 = gdf.sample(n=max_features, random_state=42).copy()
    return gdf2, msg


@st.cache_data(show_spinner=False)
def load_gdf_parquet(path_str: str, simplify_tol: float, max_features: int) -> Tuple["gpd.GeoDataFrame", Optional[str]]:
    path = Path(path_str)
    logger.info("Carregando: %s", path)
    gdf = gpd.read_parquet(path)

    # CRSs
    gdf = _ensure_crs_wgs84(gdf)

    # sanity
    gdf = _sanitize_geometries(gdf)

    # simplificação (para performance)
    gdf = _maybe_simplify(gdf, simplify_tol)

    # limite features (evitar quebrar o canvas)
    gdf, warn = _limit_features(gdf, max_features)

    return gdf, warn


def gdf_to_geojson_dict(gdf: "gpd.GeoDataFrame") -> Dict[str, Any]:
    if gdf is None or gdf.empty:
        return {"type": "FeatureCollection", "features": []}
    # IMPORTANTE: GeoJsonLayer é bem mais robusto que PathLayer para polígonos
    return json.loads(gdf.to_json())


def _compute_center_zoom(gdf: Optional["gpd.GeoDataFrame"]) -> Tuple[float, float, float]:
    # fallback centro SP
    default = (-46.6333, -23.5505, 10.5)  # lon, lat, zoom
    if gdf is None or gdf.empty:
        return default
    try:
        minx, miny, maxx, maxy = gdf.total_bounds
        lon = float((minx + maxx) / 2.0)
        lat = float((miny + maxy) / 2.0)
        # zoom heurístico simples
        span = max(maxx - minx, maxy - miny)
        if span <= 0:
            return (lon, lat, 12.0)
        zoom = float(np.clip(12.5 - np.log(span + 1e-9) * 2.0, 8.0, 14.5))
        return (lon, lat, zoom)
    except Exception:
        return default


# ============================================================
# VARIÁVEIS — Choropleth simples (evita JS quebrar)
# ============================================================
def _assign_fill_colors(gdf: "gpd.GeoDataFrame", value_col: str) -> "gpd.GeoDataFrame":
    """
    Cria uma coluna 'fill_color' (RGBA) para ser lida pelo GeoJsonLayer.
    """
    if gdf.empty or value_col not in gdf.columns:
        return gdf

    s = pd.to_numeric(gdf[value_col], errors="coerce")
    gdf = gdf.copy()

    # paleta discreta (sem depender de libs)
    palette = [
        [254, 237, 222, 120],
        [253, 208, 162, 120],
        [253, 174, 107, 120],
        [253, 141,  60, 120],
        [241, 105,  19, 120],
        [217,  72,   1, 120],
        [166,  54,   3, 120],
    ]

    finite = s[np.isfinite(s)]
    if finite.empty:
        gdf["fill_color"] = [[0, 0, 0, 0] for _ in range(len(gdf))]
        return gdf

    qs = np.quantile(finite, [0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0])
    qs = np.unique(qs)  # evita bins repetidos
    if len(qs) < 3:
        gdf["fill_color"] = [palette[-1] for _ in range(len(gdf))]
        return gdf

    # binning
    def pick_color(v: float) -> List[int]:
        if not np.isfinite(v):
            return [0, 0, 0, 0]
        # encontra intervalo
        idx = int(np.searchsorted(qs, v, side="right") - 1)
        idx = int(np.clip(idx, 0, len(palette) - 1))
        return palette[idx]

    gdf["fill_color"] = [pick_color(v) for v in s.to_numpy()]
    return gdf


# ============================================================
# REFERÊNCIAS (evita NameError + não quebra se arquivo não existe)
# ============================================================
def load_green_areas() -> Optional["gpd.GeoDataFrame"]:
    """
    Loader seguro. Se não houver arquivo, retorna None (sem quebrar o app).
    Ajuste os candidates para o nome real do seu parquet.
    """
    try:
        p = _find_parquet(PATH_REFERENCIAS, ["areas_verdes.parquet", "areas_verdes", "green_areas.parquet"])
        if p is None:
            return None
        gdf, _ = load_gdf_parquet(str(p), simplify_tol=0.00008, max_features=8000)
        return gdf
    except Exception as e:
        logger.exception("Falha ao carregar áreas verdes: %s", e)
        return None


def collect_reference_overlays() -> List["pdk.Layer"]:
    """
    Sempre retorna uma lista (possivelmente vazia) e NUNCA quebra o app.
    """
    layers: List[pdk.Layer] = []
    try:
        g = load_green_areas()
        if g is not None and not g.empty:
            gj = gdf_to_geojson_dict(g)
            layers.append(
                pdk.Layer(
                    "GeoJsonLayer",
                    data=gj,
                    id="ref-green-areas",
                    filled=True,
                    stroked=False,
                    get_fill_color=[70, 150, 70, 70],  # leve
                    pickable=False,
                )
            )
    except Exception as e:
        logger.exception("Erro em overlays de referência (seguindo sem quebrar): %s", e)

    return layers


# ============================================================
# CAMADAS — limites/variáveis em GeoJsonLayer (sem PathLayer!)
# ============================================================
def build_limites_layer(gdf: "gpd.GeoDataFrame", layer_id: str, line_color=(255, 255, 255, 220)) -> "pdk.Layer":
    gj = gdf_to_geojson_dict(gdf)
    return pdk.Layer(
        "GeoJsonLayer",
        data=gj,
        id=layer_id,
        filled=False,          # <- contorno apenas
        stroked=True,
        get_line_color=list(line_color),
        get_line_width=1.2,
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
    )


def build_choropleth_layer(gdf: "gpd.GeoDataFrame", layer_id: str) -> "pdk.Layer":
    gj = gdf_to_geojson_dict(gdf)
    # Acessor robusto: lê array RGBA vindo de properties.fill_color
    return pdk.Layer(
        "GeoJsonLayer",
        data=gj,
        id=layer_id,
        filled=True,
        stroked=True,
        get_fill_color="properties.fill_color",   # <- evita JS undefined
        get_line_color=[255, 255, 255, 70],
        get_line_width=0.5,
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
    )


# ============================================================
# RENDER PYDECK (sempre renderiza base + layers válidas)
# ============================================================
def render_pydeck(
    center_lon: float,
    center_lat: float,
    zoom: float,
    basemap_key: str,
    limite_layer: Optional["pdk.Layer"],
    var_layer: Optional["pdk.Layer"],
    show_references: bool,
) -> None:
    layers: List["pdk.Layer"] = []

    # Ordem: variável (fill) -> limites (stroke) -> refs (semi-transparent)
    if var_layer is not None:
        layers.append(var_layer)
    if limite_layer is not None:
        layers.append(limite_layer)
    if show_references:
        layers.extend(collect_reference_overlays())

    view_state = pdk.ViewState(
        longitude=center_lon,
        latitude=center_lat,
        zoom=float(zoom),
        pitch=0,
        bearing=0,
    )

    map_style = BASEMAPS.get(basemap_key, BASEMAPS["Carto Positron (leve)"])

    deck = pdk.Deck(
        map_style=map_style,   # <- base sempre definido (não “some”)
        initial_view_state=view_state,
        layers=layers,
        tooltip={"text": "{name}"} if True else None,
    )

    # key força re-render quando troca basemap/camadas
    st.pydeck_chart(deck, use_container_width=True, height=760, key=f"map::{basemap_key}::{len(layers)}")


# ============================================================
# UI / MAIN
# ============================================================
def main() -> None:
    st.title("Dash PlanBairros — Mapa")

    with st.sidebar:
        st.header("Configurações")

        basemap_key = st.selectbox("Fundo do mapa", list(BASEMAPS.keys()), index=0)

        st.divider()
        st.subheader("Limites administrativos")

        limites_on = st.toggle("Exibir limites", value=False)

        limites_opts = {
            "Distritos": ["Distritos.parquet", "Distritos"],
            "SetoresCensitarios2023": ["SetoresCensitarios2023.parquet", "SetoresCensitarios2023", "Setores_2023.parquet"],
            "ZonasOD2023": ["ZonasOD2023.parquet", "ZonasOD2023", "Zonas_OD_2023.parquet"],
            "Subprefeitura": ["Subprefeitura.parquet", "Subprefeitura", "Subprefeituras.parquet"],
        }

        limite_choice = st.selectbox("Camada", list(limites_opts.keys()), index=0, disabled=not limites_on)

        st.divider()
        st.subheader("Variáveis")

        var_on = st.toggle("Exibir variável (choropleth)", value=False)
        var_choice = st.selectbox(
            "Variável",
            ["densidade", "zoneamento"],
            index=0,
            disabled=not var_on,
        )

        st.divider()
        st.subheader("Performance")
        max_features = st.number_input(
            "Máx. polígonos por camada",
            min_value=1000,
            max_value=50000,
            value=MAX_FEATURES_DEFAULT,
            step=1000,
            help="Evita travar o navegador/Deck.gl.",
        )
        simplify_tol = st.slider(
            "Simplificação geométrica (graus)",
            min_value=0.0,
            max_value=0.0010,
            value=0.00008,
            step=0.00002,
            help="Aumente se estiver lento (perde detalhe). 0 desliga.",
        )

        st.divider()
        show_refs = st.toggle("Overlays de referência (opcional)", value=False)

    # ----------------------------
    # Carregamento das camadas
    # ----------------------------
    limite_layer = None
    var_layer = None

    # Limites
    limite_gdf = None
    limite_warn = None
    if limites_on:
        try:
            p = _find_parquet(PATH_LIMITES, limites_opts[limite_choice])
            if p is None:
                st.warning(f"Arquivo de limites não encontrado para: {limite_choice}")
            else:
                limite_gdf, limite_warn = load_gdf_parquet(str(p), simplify_tol=simplify_tol, max_features=int(max_features))
                if limite_gdf is not None and not limite_gdf.empty:
                    limite_layer = build_limites_layer(limite_gdf, layer_id=f"limite::{limite_choice}")
        except Exception as e:
            logger.exception("Erro ao carregar limites: %s", e)
            st.error("Erro ao carregar limites (veja logs).")
            st.exception(e)

    # Variáveis
    var_gdf = None
    var_warn = None
    if var_on:
        try:
            # candidates tolerantes
            p = _find_parquet(PATH_VARIAVEIS, [f"{var_choice}.parquet", var_choice])
            if p is None:
                # tenta fallback no root (caso sua pasta seja diferente)
                p = _find_parquet(ROOT, [f"{var_choice}.parquet", var_choice])
            if p is None:
                st.warning(f"Arquivo de variável não encontrado: {var_choice}")
            else:
                var_gdf, var_warn = load_gdf_parquet(str(p), simplify_tol=simplify_tol, max_features=int(max_features))

                # define qual coluna usar (tolerante)
                # ajuste aqui se suas colunas tiverem nomes específicos
                col_candidates = {
                    "densidade": ["densidade", "DENSIDADE", "dens_hab_ha", "dens_hab"],
                    "zoneamento": ["zoneamento", "ZONEAMENTO", "zona", "ZONA"],
                }
                value_col = None
                for c in col_candidates[var_choice]:
                    if c in var_gdf.columns:
                        value_col = c
                        break

                if value_col is None:
                    st.warning(f"Não encontrei coluna de valor para '{var_choice}'. Ajuste col_candidates no código.")
                else:
                    var_gdf = _assign_fill_colors(var_gdf, value_col=value_col)
                    var_layer = build_choropleth_layer(var_gdf, layer_id=f"var::{var_choice}")
        except Exception as e:
            logger.exception("Erro ao carregar variável: %s", e)
            st.error("Erro ao carregar variável (veja logs).")
            st.exception(e)

    # ----------------------------
    # Centro/zoom: prioriza variável, depois limites
    # ----------------------------
    ref_gdf = var_gdf if (var_gdf is not None and not var_gdf.empty) else limite_gdf
    lon, lat, zoom = _compute_center_zoom(ref_gdf)

    # Alertas de performance (sem quebrar)
    if limite_warn:
        st.info(limite_warn)
    if var_warn:
        st.info(var_warn)

    # ----------------------------
    # Render
    # ----------------------------
    render_pydeck(
        center_lon=lon,
        center_lat=lat,
        zoom=zoom,
        basemap_key=basemap_key,
        limite_layer=limite_layer,
        var_layer=var_layer,
        show_references=show_refs,
    )

    # Debug opcional (não pesa)
    with st.expander("Diagnóstico rápido (debug)", expanded=False):
        st.write("Basemap:", basemap_key)
        st.write("Limites ON:", limites_on, "| Camada:", limite_choice)
        st.write("Variável ON:", var_on, "| Variável:", var_choice)
        st.write("max_features:", int(max_features), "| simplify_tol:", simplify_tol)
        if limite_gdf is not None:
            st.write("Limites features:", int(len(limite_gdf)))
        if var_gdf is not None:
            st.write("Variável features:", int(len(var_gdf)))


if __name__ == "__main__":
    main()
