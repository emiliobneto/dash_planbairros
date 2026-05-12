from __future__ import annotations
import streamlit as st
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import base64
import html
import json
import re
import shutil
import requests
import pandas as pd
import time

# =============================================================================
# IMPORTS COM FALLBACK E DEBUG
# =============================================================================
@st.cache_resource
def load_geo_libs():
    """Carrega bibliotecas geo com debug detalhado"""
    try:
        import geopandas as gpd
        import folium
        from folium.features import GeoJsonTooltip
        from folium.plugins import Draw
        from streamlit_folium import st_folium
        from shapely.geometry import Point, shape
        
        st.success("✅ Geo bibliotecas carregadas!")
        return gpd, folium, GeoJsonTooltip, Draw, st_folium, Point, shape
    except ImportError as e:
        st.error(f"❌ Erro importação: {e}")
        st.info("""
        **Instale as dependências:**
        ```bash
        pip install geopandas folium streamlit-folium shapely pyarrow
        ```
        """)
        return None, None, None, None, None, None, None

gpd, folium, GeoJsonTooltip, Draw, st_folium, Point, shape = load_geo_libs()

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(
    page_title="PlanBairros ✅", page_icon="🗺️", layout="wide",
    initial_sidebar_state="collapsed"
)

PB_COLORS = {"navy": "#14407D", "teal": "#1C6880", "brown": "#C65534"}
PB_NAVY, PB_BTN = PB_COLORS["navy"], PB_COLORS["teal"]

# 🗺️ BASEMAPS CONFIÁVEIS (CORRIGIDO)
BASEMAPS = {
    "osm": {
        "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attr": "© OpenStreetMap contributors"
    },
    "hot": {
        "url": "https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png", 
        "attr": "© OpenStreetMap contributors"
    },
    "stamen": {
        "url": "https://stamen-tiles-{s}.a.ssl.fastly.net/toner-lite/{z}/{x}/{y}{r}.png",
        "attr": "© Stamen Design"
    }
}

# =============================================================================
# PATHS E CACHE
# =============================================================================
REPO_ROOT = Path.cwd()
DATA_CACHE_DIR = REPO_ROOT / "data_cache"
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR = REPO_ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "logo_todos.jpg"

# IDS PADRONIZADOS
SUBPREF_ID, DIST_ID, ISO_ID, QUADRA_ID, CENSO_ID = "subpref_id", "distrito_id", "iso_id", "quadra_id", "censo_id"
QUADRA_UID = "quadra_uid"

# LINKS DIRETOS CORRIGIDOS (FILE_IDs extraídos)
LAYER_DRIVE_IDS = {
    "subpref": "1vPY34cQLCoGfADpyOJjL9pNCYkVrmSZA",
    "dist": "1K-t2BiSHN_D8De0oCFxzGdrEMhnGnh10", 
    "iso": "1rSTVu_i-z07vKLbG3ElUNchWvvKih3xJ",
    "censo": "1APp7fxT2mgTpegVisVyQwjTRWOPz6Rgn",
    "od": "18yFCikpYxSvH8sqh8qULq-nMFRo2CqL7",
    "quadra": "1Ivy2PyGHqFgIxSMoK3N9oik2wr5v912U"
}

LOCAL_FILENAMES = {
    "subpref": "Subprefeitura.parquet", "dist": "Distrito.parquet",
    "iso": "Isocronas.parquet", "censo": "Setorcensitario.parquet",
    "od": "ZonasOD.parquet", "quadra": "Quadras.parquet"
}

# =============================================================================
# CSS E HEADER
# =============================================================================
def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700;900&display=swap');
    .stApp {{ font-family: 'Roboto', sans-serif; }}
    .main .block-container {{ padding-top: 1rem; }}
    .pb-header {{ 
        background: linear-gradient(135deg, {PB_NAVY}, {PB_BTN});
        color: white; border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
    }}
    .pb-title {{ font-size: 2.5rem; font-weight: 900; }}
    .pb-card {{ 
        background: white; border-radius: 16px; 
        box-shadow: 0 10px 30px rgba(0,0,0,0.1); padding: 1.5rem;
    }}
    .debug-panel {{ background: #f8f9fa; border-left: 4px solid {PB_BTN}; }}
    </style>
    """, unsafe_allow_html=True)

def render_header():
    logo = "https://raw.githubusercontent.com/streamlit/brand/main/logomark/streamlit-mark-color.png"
    if LOGO_PATH.exists():
        with open(LOGO_PATH, "rb") as f:
            logo = base64.b64encode(f.read()).decode()
            logo = f"data:image/jpeg;base64,{logo}"
    
    st.markdown(f"""
    <div class="pb-header">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <img src="{logo}" style="height: 50px; border-radius: 8px;">
            <div>
                <div class="pb-title">🗺️ PlanBairros</div>
                <div style="opacity: 0.9;">Plataforma de planejamento urbano - SP</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# DOWNLOAD ROBUSTO (CORRIGIDO)
# =============================================================================
def download_drive_file_robust(file_id: str, dst: Path, label: str) -> bool:
    """Download otimizado com múltiplos fallbacks"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    if dst.exists() and dst.stat().st_size > 10_000:  # >10KB válido
        return True
    
    urls = [        f"https://drive.google.com/uc?export=download&id={file_id}",
        f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
        f"https://drive.google.com/uc?id={file_id}&export=download"
    ]
    
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    prog = st.progress(0, text=f"📥 {label}...")
    
    for i, url in enumerate(urls):
        try:
            resp = session.get(url, stream=True, timeout=60, allow_redirects=True)
            
            if resp.status_code == 200:
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                
                with open(dst, "wb") as f:
                    for chunk in resp.iter_content(1024*1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                prog.progress(min(downloaded/total, 1), 
                                            text=f"📥 {label}... {downloaded/1e6:.1f}MB")
                
                if dst.stat().st_size > 10_000:
                    prog.empty()
                    st.success(f"✅ {label} baixado!")
                    return True
            
        except Exception:
            pass
    
    prog.empty()
    st.error(f"❌ Falha {label}")
    return False

def ensure_layer(layer_key: str) -> Path:
    """Garante layer local com download robusto"""
    dst = DATA_CACHE_DIR / LOCAL_FILENAMES[layer_key]
    
    if dst.exists() and dst.stat().st_size > 10_000:
        return dst
    
    file_id = LAYER_DRIVE_IDS.get(layer_key)
    if not file_id:
        st.error(f"❌ ID não encontrado: {layer_key}")
        return dst
    
    success = download_drive_file_robust(file_id, dst, LOCAL_FILENAMES[layer_key])
    return dst if success else Path()

# =============================================================================
# LEITURA OTIMIZADA
# =============================================================================
@st.cache_data(ttl=3600)
def read_layer_cached(layer_key: str) -> Optional['gpd.GeoDataFrame']:
    if gpd is None:
        return None
    
    path = ensure_layer(layer_key)
    if not path.exists():
        return None
    
    try:
        gdf = gpd.read_parquet(path)
        if gdf.empty or gdf.geometry.isna().all():
            return None
        
        # Padronizar CRS
        if gdf.crs is None:
            gdf.set_crs(4326, inplace=True)
        else:
            gdf = gdf.to_crs(4326)
            
        st.success(f"✅ {layer_key}: {len(gdf):,} features")
        return gdf
        
    except Exception as e:
        st.error(f"❌ Erro leitura {layer_key}: {e}")
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
        "view_center": (-23.55, -46.63),
        "view_zoom": 11,
        "debug_mode": True
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# =============================================================================
# MAPA FOLIUM (TOTALMENTE RECRIADO)
# =============================================================================
def create_map(center=(-23.55, -46.63), zoom=11):
    """Mapa com basemaps confiáveis"""
    if folium is None:
        return None
    
    m = folium.Map(
        location=center, zoom_start=zoom,
        tiles=None, control_scale=True, prefer_canvas=True
    )
    
    # Adicionar múltiplos basemaps
    folium.TileLayer(
        tiles=BASEMAPS["osm"]["url"],
        attr=BASEMAPS["osm"]["attr"],
        name="OpenStreetMap",
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles=BASEMAPS["stamen"]["url"], 
        attr=BASEMAPS["stamen"]["attr"],
        name="Stamen Toner",
        control=True
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    return m

def add_layer_to_map(m, gdf, name, id_col, color="#1f77b4", weight=2):
    """Adiciona layer com tooltip e seleção"""
    if m is None or gdf is None or gdf.empty:
        return
    
    # Tooltip
    tooltip = f"{name}: " + gdf[id_col].astype(str)
    
    folium.GeoJson(
        gdf.__geo_interface__,
        name=name,
        style_function=lambda x: {
            'fillColor': color, 'color': 'black', 'weight': weight,
            'fillOpacity': 0.3
        },
        tooltip=folium.GeoJsonTooltip(fields=[id_col], aliases=[name])
    ).add_to(m)
    
    folium.LayerControl().add_to(m)

# =============================================================================
# UI PRINCIPAL
# =============================================================================
def debug_panel():
    """Painel de debug com status"""
    with st.sidebar:
        st.markdown("### 🔧 Debug")
        
        # Status cache
        files = {k: ensure_layer(k).exists() for k in LAYER_DRIVE_IDS}
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Arquivos OK", sum(files.values()), delta=len(files))
        with col2:
            st.metric("Cache", f"{DATA_CACHE_DIR.stat().st_size/1e6:.1f}MB")
        
        st.markdown("**Status por layer:**")
        for layer, ok in files.items():
            st.caption(f"{'✅' if ok else '❌'} {layer}")
        
        if st.button("🧹 Limpar Cache"):
            shutil.rmtree(DATA_CACHE_DIR, ignore_errors=True)
            DATA_CACHE_DIR.mkdir()
            st.rerun()
        
        st.markdown("---")
        if st.button("🔄 Recarregar Tudo"):
            st.cache_data.clear()
            st.rerun()

def render_level(level):
    """Renderiza mapa por nível com dados reais"""
    m = create_map(st.session_state["view_center"], st.session_state["view_zoom"])
    
    if level == "subpref":
        gdf = read_layer_cached("subpref")
        if gdf is not None:
            add_layer_to_map(m, gdf, "Subprefeituras", SUBPREF_ID, "#FF7F0E")
        st.markdown("### 🏛️ **Subprefeituras**")
    
    elif level == "distrito":
        gdf_dist = read_layer_cached("dist")
        gdf_sub = read_layer_cached("subpref")
        if gdf_dist is not None:
            add_layer_to_map(m, gdf_dist, "Distritos", DIST_ID, "#2CA02C")
        st.markdown("### 🏘️ **Distritos**")
    
    elif level == "iso":
        gdf = read_layer_cached("iso")
        if gdf is not None:
            add_layer_to_map(m, gdf, "Isócronas", ISO_ID, "#D62728")
        st.markdown("### ⏱️ **Isócronas**")
    
    else:  # quadra
        gdf = read_layer_cached("quadra")
        if gdf is not None:
            add_layer_to_map(m, gdf, "Quadras", QUADRA_ID, "#9467BD")
        st.markdown("### 🏠 **Quadras**")
    
    if m and st_folium:
        map_data = st_folium(m, height=700, width=1000, key="main_map")
        st.session_state["view_center"] = map_data.get("last_clicked", st.session_state["view_center"])
    
    else:
        st.error("❌ Mapa não carregou. Verifique debug panel →")

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    inject_css()
    render_header()
    init_state()
    
    # Debug sempre visível
    debug_panel()
    
    # Controle de nível
    col1, col2 = st.columns([3, 1])
    
    with col1:
        level = st.selectbox("**Nível**", 
                           ["subpref", "distrito", "iso", "quadra"],
                           format_func={
                               "subpref": "🏛️ Subprefeituras",
                               "distrito": "🏘️ Distritos", 
                               "iso": "⏱️ Isócronas",
                               "quadra": "🏠 Quadras"
                           },
                           key="level_select")
    
    with col2:
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("⬅️ Voltar", use_container_width=True):
                levels = ["subpref", "distrito", "iso", "quadra"]
                idx = levels.index(level) - 1
                if idx >= 0:
                    st.session_state["level"] = levels[idx]
                    st.rerun()
        with col_btn2:
            if st.button("🔄 Reset", use_container_width=True):
                for k in ["selected_subpref_id", "selected_distrito_id", "selected_iso_ids"]:
                    st.session_state[k] = None if "id" in k else set()
                st.rerun()
    
    st.divider()
    
    # Render mapa
    with st.container():
        st.markdown(f"<div class='pb-card'>", unsafe_allow_html=True)
        render_level(st.session_state["level"])
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Info final
    st.markdown("---")
    st.caption("👨‍💻 PlanBairros v2.0 - Correções aplicadas: Drive robusto + Basemaps + Cache intel.")

if __name__ == "__main__":
    main()
