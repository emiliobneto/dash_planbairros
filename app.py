from __future__ import annotations

import base64
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import streamlit as st


# =============================================================================
# CONFIG STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="PlanBairros ✅",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =============================================================================
# IMPORTS COM FALLBACK E DEBUG
# =============================================================================
@st.cache_resource
def load_geo_libs():
    """Carrega bibliotecas geográficas com fallback."""
    try:
        import geopandas as gpd
        import folium
        from folium.features import GeoJsonTooltip
        from folium.plugins import Draw
        from streamlit_folium import st_folium
        from shapely.geometry import Point, shape

        return gpd, folium, GeoJsonTooltip, Draw, st_folium, Point, shape

    except ImportError as e:
        st.error(f"❌ Erro ao importar bibliotecas geográficas: {e}")
        st.info(
            """
            **Instale as dependências:**

            ```bash
            pip install geopandas folium streamlit-folium shapely pyarrow
            ```
            """
        )
        return None, None, None, None, None, None, None


gpd, folium, GeoJsonTooltip, Draw, st_folium, Point, shape = load_geo_libs()


# =============================================================================
# CORES E CONSTANTES
# =============================================================================
PB_COLORS = {
    "navy": "#14407D",
    "teal": "#1C6880",
    "brown": "#C65534",
}

PB_NAVY = PB_COLORS["navy"]
PB_BTN = PB_COLORS["teal"]

LEVELS = ["subpref", "distrito", "iso", "quadra"]

LEVEL_LABELS = {
    "subpref": "🏛️ Subprefeituras",
    "distrito": "🏘️ Distritos",
    "iso": "⏱️ Isócronas",
    "quadra": "🏠 Quadras",
}

BASEMAPS = {
    "osm": {
        "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attr": "© OpenStreetMap contributors",
        "name": "OpenStreetMap",
    },
    "hot": {
        "url": "https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        "attr": "© OpenStreetMap contributors, Humanitarian OpenStreetMap Team",
        "name": "OpenStreetMap HOT",
    },
    "cartodb": {
        "url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        "attr": "© OpenStreetMap contributors © CARTO",
        "name": "CartoDB Positron",
    },
}


# =============================================================================
# PATHS E CACHE
# =============================================================================
REPO_ROOT = Path.cwd()

DATA_CACHE_DIR = REPO_ROOT / "data_cache"
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ASSETS_DIR = REPO_ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "logo_todos.jpg"


# =============================================================================
# IDS PADRONIZADOS
# =============================================================================
SUBPREF_ID = "subpref_id"
DIST_ID = "distrito_id"
ISO_ID = "iso_id"
QUADRA_ID = "quadra_id"
CENSO_ID = "censo_id"
QUADRA_UID = "quadra_uid"


# =============================================================================
# GOOGLE DRIVE IDS
# =============================================================================
LAYER_DRIVE_IDS = {
    "subpref": "1vPY34cQLCoGfADpyOJjL9pNCYkVrmSZA",
    "dist": "1K-t2BiSHN_D8De0oCFxzGdrEMhnGnh10",
    "iso": "1rSTVu_i-z07vKLbG3ElUNchWvvKih3xJ",
    "censo": "1APp7fxT2mgTpegVisVyQwjTRWOPz6Rgn",
    "od": "18yFCikpYxSvH8sqh8qULq-nMFRo2CqL7",
    "quadra": "1Ivy2PyGHqFgIxSMoK3N9oik2wr5v912U",
}

LOCAL_FILENAMES = {
    "subpref": "Subprefeitura.parquet",
    "dist": "Distrito.parquet",
    "iso": "Isocronas.parquet",
    "censo": "Setorcensitario.parquet",
    "od": "ZonasOD.parquet",
    "quadra": "Quadras.parquet",
}


# =============================================================================
# CSS E HEADER
# =============================================================================
def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700;900&display=swap');

        .stApp {{
            font-family: 'Roboto', sans-serif;
        }}

        .main .block-container {{
            padding-top: 1rem;
        }}

        .pb-header {{
            background: linear-gradient(135deg, {PB_NAVY}, {PB_BTN});
            color: white;
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }}

        .pb-title {{
            font-size: 2.5rem;
            font-weight: 900;
        }}

        .pb-card {{
            background: white;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            padding: 1.5rem;
        }}

        .debug-panel {{
            background: #f8f9fa;
            border-left: 4px solid {PB_BTN};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    logo = "https://raw.githubusercontent.com/streamlit/brand/main/logomark/streamlit-mark-color.png"

    if LOGO_PATH.exists():
        with open(LOGO_PATH, "rb") as f:
            logo = base64.b64encode(f.read()).decode()
            logo = f"data:image/jpeg;base64,{logo}"

    st.markdown(
        f"""
        <div class="pb-header">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <img src="{logo}" style="height: 50px; border-radius: 8px;">
                <div>
                    <div class="pb-title">🗺️ PlanBairros</div>
                    <div style="opacity: 0.9;">Plataforma de planejamento urbano - SP</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# UTILITÁRIOS
# =============================================================================
def get_dir_size(path: Path) -> int:
    """Retorna tamanho total de uma pasta em bytes."""
    if not path.exists():
        return 0

    total = 0

    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass

    return total


def is_valid_file(path: Path, min_size: int = 10_000) -> bool:
    """Verifica se arquivo existe e tem tamanho mínimo."""
    return path.exists() and path.is_file() and path.stat().st_size > min_size


def get_layer_path(layer_key: str) -> Path:
    """Retorna o caminho local esperado de uma camada."""
    filename = LOCAL_FILENAMES.get(layer_key)

    if not filename:
        return Path()

    return DATA_CACHE_DIR / filename


# =============================================================================
# DOWNLOAD ROBUSTO
# =============================================================================
def download_drive_file_robust(file_id: str, dst: Path, label: str) -> bool:
    """Download de arquivo do Google Drive com múltiplos fallbacks."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    if is_valid_file(dst):
        return True

    urls = [
        f"https://drive.google.com/uc?export=download&id={file_id}",
        f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
        f"https://drive.google.com/uc?id={file_id}&export=download",
    ]

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        }
    )

    progress = st.progress(0, text=f"📥 Baixando {label}...")

    for url in urls:
        try:
            response = session.get(
                url,
                stream=True,
                timeout=60,
                allow_redirects=True,
            )

            if response.status_code != 200:
                continue

            total = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(dst, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total > 0:
                            progress.progress(
                                min(downloaded / total, 1.0),
                                text=f"📥 Baixando {label}... {downloaded / 1e6:.1f} MB",
                            )

            if is_valid_file(dst):
                progress.empty()
                st.success(f"✅ {label} baixado com sucesso!")
                return True

        except Exception:
            continue

    progress.empty()
    st.error(f"❌ Falha ao baixar {label}")
    return False


def ensure_layer(layer_key: str) -> Path:
    """Garante que a camada esteja disponível localmente."""
    dst = get_layer_path(layer_key)

    if not dst:
        st.error(f"❌ Nome local não encontrado para camada: {layer_key}")
        return Path()

    if is_valid_file(dst):
        return dst

    file_id = LAYER_DRIVE_IDS.get(layer_key)

    if not file_id:
        st.error(f"❌ ID do Google Drive não encontrado para camada: {layer_key}")
        return Path()

    success = download_drive_file_robust(
        file_id=file_id,
        dst=dst,
        label=LOCAL_FILENAMES[layer_key],
    )

    return dst if success else Path()


# =============================================================================
# LEITURA OTIMIZADA
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def read_layer_cached(layer_key: str):
    """Lê uma camada parquet como GeoDataFrame."""
    if gpd is None:
        return None

    path = ensure_layer(layer_key)

    if not path.exists():
        return None

    try:
        gdf = gpd.read_parquet(path)

        if gdf.empty:
            return None

        if "geometry" not in gdf.columns:
            st.error(f"❌ Camada {layer_key} não possui coluna geometry.")
            return None

        if gdf.geometry.isna().all():
            st.error(f"❌ Camada {layer_key} possui geometria vazia.")
            return None

        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        else:
            gdf = gdf.to_crs(4326)

        return gdf

    except Exception as e:
        st.error(f"❌ Erro ao ler camada {layer_key}: {e}")
        return None


# =============================================================================
# STATE MANAGEMENT
# =============================================================================
def init_state():
    defaults = {
        "level": "subpref",
        "selected_subpref_id": None,
        "selected_distrito_id": None,
        "selected_iso_ids": set(),
        "view_center": [-23.55, -46.63],
        "view_zoom": 11,
        "debug_mode": True,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# MAPA FOLIUM
# =============================================================================
def create_map(center=None, zoom: int = 11):
    """Cria mapa Folium com basemaps confiáveis."""
    if folium is None:
        return None

    if center is None:
        center = [-23.55, -46.63]

    if isinstance(center, dict):
        center = [
            center.get("lat", -23.55),
            center.get("lng", -46.63),
        ]

    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    for basemap in BASEMAPS.values():
        folium.TileLayer(
            tiles=basemap["url"],
            attr=basemap["attr"],
            name=basemap["name"],
            control=True,
        ).add_to(m)

    return m


def add_layer_to_map(
    m,
    gdf,
    name: str,
    id_col: str,
    color: str = "#1f77b4",
    weight: int = 2,
):
    """Adiciona GeoDataFrame ao mapa."""
    if m is None or gdf is None or gdf.empty:
        return

    gdf_plot = gdf.copy()

    if id_col not in gdf_plot.columns:
        gdf_plot[id_col] = gdf_plot.index.astype(str)

    folium.GeoJson(
        data=gdf_plot.__geo_interface__,
        name=name,
        style_function=lambda feature: {
            "fillColor": color,
            "color": "black",
            "weight": weight,
            "fillOpacity": 0.3,
        },
        highlight_function=lambda feature: {
            "fillColor": color,
            "color": "#000000",
            "weight": weight + 1,
            "fillOpacity": 0.55,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[id_col],
            aliases=[name],
            localize=True,
            sticky=True,
        ),
    ).add_to(m)


# =============================================================================
# UI PRINCIPAL
# =============================================================================
def debug_panel():
    """Painel lateral de debug."""
    with st.sidebar:
        st.markdown("### 🔧 Debug")

        files = {}

        for layer_key in LAYER_DRIVE_IDS:
            layer_path = get_layer_path(layer_key)
            files[layer_key] = is_valid_file(layer_path)

        cache_size_mb = get_dir_size(DATA_CACHE_DIR) / 1e6

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Arquivos OK", sum(files.values()))

        with col2:
            st.metric("Cache", f"{cache_size_mb:.1f} MB")

        st.markdown("**Status por layer:**")

        for layer, ok in files.items():
            st.caption(f"{'✅' if ok else '❌'} {layer}")

        st.markdown("---")

        if st.button("📥 Baixar arquivos faltantes", use_container_width=True):
            for layer_key in LAYER_DRIVE_IDS:
                ensure_layer(layer_key)

            st.rerun()

        if st.button("🧹 Limpar Cache", use_container_width=True):
            shutil.rmtree(DATA_CACHE_DIR, ignore_errors=True)
            DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            st.cache_data.clear()
            st.rerun()

        if st.button("🔄 Recarregar Tudo", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()


def render_level(level: str):
    """Renderiza mapa conforme o nível selecionado."""
    m = create_map(
        center=st.session_state["view_center"],
        zoom=st.session_state["view_zoom"],
    )

    if level == "subpref":
        st.markdown("### 🏛️ **Subprefeituras**")
        gdf = read_layer_cached("subpref")

        if gdf is not None:
            add_layer_to_map(
                m=m,
                gdf=gdf,
                name="Subprefeituras",
                id_col=SUBPREF_ID,
                color="#FF7F0E",
            )
            st.caption(f"✅ {len(gdf):,} feições carregadas.")

    elif level == "distrito":
        st.markdown("### 🏘️ **Distritos**")
        gdf = read_layer_cached("dist")

        if gdf is not None:
            add_layer_to_map(
                m=m,
                gdf=gdf,
                name="Distritos",
                id_col=DIST_ID,
                color="#2CA02C",
            )
            st.caption(f"✅ {len(gdf):,} feições carregadas.")

    elif level == "iso":
        st.markdown("### ⏱️ **Isócronas**")
        gdf = read_layer_cached("iso")

        if gdf is not None:
            add_layer_to_map(
                m=m,
                gdf=gdf,
                name="Isócronas",
                id_col=ISO_ID,
                color="#D62728",
            )
            st.caption(f"✅ {len(gdf):,} feições carregadas.")

    elif level == "quadra":
        st.markdown("### 🏠 **Quadras**")
        gdf = read_layer_cached("quadra")

        if gdf is not None:
            add_layer_to_map(
                m=m,
                gdf=gdf,
                name="Quadras",
                id_col=QUADRA_ID,
                color="#9467BD",
                weight=1,
            )
            st.caption(f"✅ {len(gdf):,} feições carregadas.")

    else:
        st.warning("⚠️ Nível inválido selecionado.")

    if m is not None and folium is not None:
        folium.LayerControl(collapsed=False).add_to(m)

    if m is not None and st_folium is not None:
        map_data = st_folium(
            m,
            height=700,
            width=None,
            key=f"main_map_{level}",
            returned_objects=["last_clicked", "center", "zoom"],
        )

        if map_data:
            center = map_data.get("center")
            zoom = map_data.get("zoom")

            if isinstance(center, dict) and "lat" in center and "lng" in center:
                st.session_state["view_center"] = [
                    center["lat"],
                    center["lng"],
                ]

            if isinstance(zoom, int):
                st.session_state["view_zoom"] = zoom

    else:
        st.error("❌ Mapa não carregou. Verifique as dependências no painel de debug.")


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    inject_css()
    render_header()
    init_state()

    debug_panel()

    col1, col2 = st.columns([3, 1])

    with col1:
        current_level = st.session_state.get("level", "subpref")

        if current_level not in LEVELS:
            current_level = "subpref"
            st.session_state["level"] = current_level

        current_index = LEVELS.index(current_level)

        level = st.selectbox(
            "**Nível**",
            LEVELS,
            index=current_index,
            format_func=lambda x: LEVEL_LABELS.get(x, x),
            key="level_select",
        )

        if level != st.session_state["level"]:
            st.session_state["level"] = level
            st.rerun()

    with col2:
        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("⬅️ Voltar", use_container_width=True):
                current_level = st.session_state.get("level", "subpref")

                if current_level in LEVELS:
                    idx = LEVELS.index(current_level)

                    if idx > 0:
                        st.session_state["level"] = LEVELS[idx - 1]
                        st.rerun()

        with col_btn2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state["level"] = "subpref"
                st.session_state["selected_subpref_id"] = None
                st.session_state["selected_distrito_id"] = None
                st.session_state["selected_iso_ids"] = set()
                st.session_state["view_center"] = [-23.55, -46.63]
                st.session_state["view_zoom"] = 11
                st.rerun()

    st.divider()

    with st.container():
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        render_level(st.session_state["level"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        "👨‍💻 PlanBairros v2.0 - Correções aplicadas: "
        "selectbox corrigido, Drive robusto, basemaps, cache e estado do mapa."
    )


if __name__ == "__main__":
    main()
