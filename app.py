# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, List
from unicodedata import normalize as _ud_norm
import re

import pandas as pd
import plotly.express as px
import streamlit as st

# Geo libs (opcionais – o app funciona sem, só não mostra o mapa)
try:
    import geopandas as gpd  # type: ignore
    import folium  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
    import branca  # type: ignore
except Exception:
    gpd = None  # type: ignore
    folium = None  # type: ignore
    st_folium = None  # type: ignore
    branca = None  # type: ignore


# ============================================================================
# Config da página e paleta
# ============================================================================
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

# gradientes (sem azul e sem verde, conforme pedido)
GRADIENTE_BASE = ["#F4DD63", "#D58243", "#C65534"]  # amarelo → laranja → telha

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
            :root {{
                --pb-amarelo: {PB_COLORS['amarelo']};
                --pb-laranja: {PB_COLORS['laranja']};
                --pb-telha:   {PB_COLORS['telha']};
                --pb-teal:    {PB_COLORS['teal']};
                --pb-navy:    {PB_COLORS['navy']};
            }}
            html, body, .stApp {{
                font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
            }}
            .pb-header {{
                background: var(--pb-navy); color: #fff; border-radius: 18px;
                padding: 20px 24px; min-height: 110px; display: flex; align-items: center; gap: 16px;
            }}
            .pb-title {{ font-size: 2.6rem; line-height: 1.15; font-weight: 700; letter-spacing: .5px; }}
            .pb-subtitle {{ opacity: .95; margin-top: 4px; font-size: 1.2rem; }}

            .pb-card {{
                background:#fff; border:1px solid rgba(20,64,125,.10); box-shadow:0 1px 2px rgba(0,0,0,.04);
                border-radius:16px; padding:18px;
            }}
            .pb-card h4 {{ font-size: 1.3rem !important; margin: 0 0 .6rem 0; }}
            .pb-card label, .pb-card .stMarkdown p {{ font-size: 1rem !important; }}
            .pb-card div[role="combobox"] {{ font-size: 1rem !important; min-height: 44px !important; }}
            .pb-card [data-baseweb="select"] * {{ font-size: 1rem !important; }}
            .pb-card .stSelectbox svg {{ transform: scale(1.2); }}

            .stTabs [data-baseweb="tab-list"] button[role="tab"] {{
                background:transparent; border-bottom:3px solid transparent; font-weight:700; font-size: 1rem;
            }}
            .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
                border-bottom:3px solid var(--pb-teal) !important; color:var(--pb-navy) !important;
            }}

            .main .block-container {{ padding-top: .6rem; padding-bottom: .6rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def get_logo_path() -> Optional[str]:
    for p in [
        "assets/logo_todos.jpg",
        "assets/logo_paleta.jpg",
        "logo_todos.jpg",
        "logo_paleta.jpg",
        "/mnt/data/logo_todos.jpg",
        "/mnt/data/logo_paleta.jpg",
    ]:
        if Path(p).exists():
            return p
    return None


# ============================================================================
# Caminhos e loaders
# ============================================================================
REPO_ROOT = Path(__file__).resolve().parent

def _resolve_dir(subdir: str) -> Path:
    for base in (REPO_ROOT, REPO_ROOT / "dash_planbairros"):
        p = base / subdir
        if p.exists():
            return p
    return REPO_ROOT / subdir

ADM_DIR = _resolve_dir("limites_administrativos")
DENS_DIR = _resolve_dir("densidade")  # aqui ficam dados dos setores (valores por setor)
ISO_DIR  = ADM_DIR                    # isócronas também foram vistas nessa pasta

def _slug(s: str) -> str:
    s2 = _ud_norm("NFKD", str(s)).encode("ASCII", "ignore").decode("ASCII")
    return re.sub(r"[^a-z0-9]+", "", s2.strip().lower())

def _first_parquet_matching(folder: Path, names: list[str]) -> Optional[Path]:
    if not folder.exists():
        return None
    wanted = {_slug(n) for n in names}
    for fp in folder.glob("*.parquet"):
        if _slug(fp.stem) in wanted:
            return fp
    return None

ADMIN_NAME_MAP = {
    "Distritos": ["Distritos"],
    "ZonasOD2023": ["ZonasOD2023", "ZonasOD"],
    "Subprefeitura": ["Subprefeitura", "Subprefeituras", "subprefeitura"],
    # Setores não entram mais em limites (requisito 1)
}
ISOCRONAS_CANDIDATES = ["isocronas", "isócronas"]

SETORES_NAME = ["SetoresCensitarios2023", "SetoresCensitarios"]  # usado só nas variáveis

@st.cache_data(show_spinner=False)
def _safe_read_parquet(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.stat().st_size < 1024:
            return None
    except Exception:
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_admin_layer(layer_name: str):
    """Lê limites administrativos (sem setores)."""
    if gpd is None:
        st.info("Geopandas não disponível — instale `geopandas`, `shapely` e `pyarrow`.")
        return None
    path = _first_parquet_matching(ADM_DIR, ADMIN_NAME_MAP.get(layer_name, [layer_name]))
    if path is None:
        st.warning(f"Arquivo Parquet não encontrado para '{layer_name}' em {ADM_DIR}.")
        return None
    gdf = gpd.read_parquet(path)
    try:
        gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)
    except Exception:
        pass
    return gdf

@st.cache_data(show_spinner=False)
def load_setores() -> Optional[gpd.GeoDataFrame]:
    """Geometrias dos Setores Censitários (usadas apenas nas variáveis)."""
    if gpd is None:
        return None
    path = _first_parquet_matching(ADM_DIR, SETORES_NAME) or _first_parquet_matching(DENS_DIR, SETORES_NAME)
    if path is None:
        st.warning("Arquivo de Setores Censitários não encontrado (usado para variáveis).")
        return None
    gdf = gpd.read_parquet(path)
    try:
        gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)
    except Exception:
        pass
    return gdf

@st.cache_data(show_spinner=False)
def load_isocronas() -> Optional[gpd.GeoDataFrame]:
    """Isócronas (área de influência de bairro) – necessária em limites e variáveis."""
    if gpd is None:
        return None
    path = _first_parquet_matching(ISO_DIR, ISOCRONAS_CANDIDATES)
    if path is None:
        # procurar também por nome exato
        path = (ISO_DIR / "isócronas.parquet") if (ISO_DIR / "isócronas.parquet").exists() else None
    if path is None:
        return None
    gdf = gpd.read_parquet(path)
    try:
        gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)
    except Exception:
        pass
    return gdf

@st.cache_data(show_spinner=False)
def load_setores_values() -> Optional[pd.DataFrame]:
    """
    Carrega um parquet na pasta 'densidade' (ou 'limites_administrativos')
    que contenha pelo menos uma destas colunas:
      - populacao
      - densidade_demografica
      - diferenca_elevacao
      - elevacao
      - cluster (opcional)
      - CD_GEOCODI / CD_SETOR (chave)
    """
    search: List[Path] = []
    for base in (DENS_DIR, ADM_DIR):
        if base.exists():
            search += list(base.glob("*.parquet"))
    for p in search:
        df = _safe_read_parquet(p)
        if df is None:
            continue
        cols = {c.lower() for c in df.columns}
        if {"cd_geocodi", "cd_setor"}.intersection(cols) and (
            {"populacao", "densidade_demografica", "diferenca_elevacao", "elevacao"}.intersection(cols)
            or "cluster" in cols
        ):
            return df
    st.info("Não encontrei arquivo com valores por Setor (população/densidade/elevacao/cluster).")
    return None


# ============================================================================
# UI
# ============================================================================
def build_tabs() -> None:
    a1, a2, a3, a4 = st.tabs(["Aba 1", "Aba 2", "Aba 3", "Aba 4"])
    for i, aba in enumerate([a1, a2, a3, a4], start=1):
        with aba:
            st.markdown(f"**Conteúdo da Aba {i}** — espaço para textos/explicações.")
    st.write("")

def build_left_controls(key_prefix="main_") -> Tuple[str, str, bool]:
    st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
    st.markdown("### Variáveis (Setores Censitários e Isócronas)", unsafe_allow_html=True)

    # Variáveis disponíveis
    var_labels = {
        "populacao": "População (Pessoa/hec)",
        "densidade_demografica": "Densidade demográfica (hab/he)",
        "diferenca_elevacao": "Variação de elevação média",
        "elevacao": "Elevação média",
        "isocronas": "Área de influência de bairro (isócronas)",
        "cluster": "Cluster (tipologia urbana)",
    }
    var_choice = st.selectbox(
        "Selecione a variável",
        list(var_labels.values()),
        index=0,
        key=f"{key_prefix}var_choice",
    )

    st.write("")  # separador pequeno
    st.markdown("### Configurações", unsafe_allow_html=True)

    limite_choice = st.selectbox(
        "Limites Administrativos",
        ["Distritos", "ZonasOD2023", "Subprefeitura", "Isócronas (área de influência)"],
        index=0,
        key=f"{key_prefix}limite",
        help="Setores NÃO aparecem aqui; são usados apenas para variáveis.",
    )

    labels_on = st.checkbox(
        "Rótulos permanentes (dinâmicos por zoom)  ",
        value=False,
        key=f"{key_prefix}labels_on",
        help="Mostra nomes fixos nos centróides (tamanho varia com o zoom).",
    )

    st.markdown("</div>", unsafe_allow_html=True)
    # retorna chave interna da variável
    inv_map = {v: k for k, v in var_labels.items()}
    return inv_map[var_choice], limite_choice, labels_on


# ============================================================================
# Folium e desenho
# ============================================================================
SIMPLIFY_TOL = 0.0005  # ~55m

def make_satellite_map(center=(-23.55, -46.63), zoom=10, tiles_opacity=0.5):
    if folium is None:
        st.error("Instale `folium` e `streamlit-folium` para exibir o mapa.")
        return None
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    try:
        folium.map.CustomPane("vectors", z_index=650).add_to(m)
    except Exception:
        pass
    # Esri Imagery
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri Satellite",
        overlay=False,
        control=False,
        opacity=tiles_opacity,
    ).add_to(m)
    return m

def inject_label_scaler(m, min_px=12, max_px=22, min_zoom=9, max_zoom=18):
    """Redimensiona '.pb-static-label' conforme zoom do Leaflet."""
    if folium is None:
        return
    from folium import Element  # type: ignore
    map_var = m.get_name()
    js = f"""
    <script>
      (function() {{
        var minZ = {min_zoom}, maxZ = {max_zoom};
        var minPx = {min_px}, maxPx = {max_px};
        function scaleFont(z) {{
          if (z < minZ) z = minZ;
          if (z > maxZ) z = maxZ;
          var t = (z - minZ) / (maxZ - minZ);
          var px = Math.round(minPx + t * (maxPx - minPx));
          document.querySelectorAll('.pb-static-label').forEach(function(el) {{
            el.style.fontSize = px + 'px';
          }});
        }}
        {map_var}.on('zoomend', function() {{ scaleFont({map_var}.getZoom()); }});
        {map_var}.whenReady(function() {{ scaleFont({map_var}.getZoom()); }});
      }})();
    </script>
    """
    m.get_root().html.add_child(Element(js))

def add_centroid_labels(m, gdf, label_col: str):
    """Rótulos permanentes; tamanho será controlado por inject_label_scaler."""
    if folium is None:
        return
    try:
        centers = gdf.copy().to_crs(4326).representative_point()
    except Exception:
        return
    for name, pt in zip(gdf[label_col].astype(str), centers):
        if pt is None or pt.is_empty:
            continue
        html = (
            "<div class='pb-static-label' "
            "style=\"font: 600 12px/1.1 Roboto, -apple-system, Segoe UI, Helvetica, Arial, sans-serif;"
            "color:#111; text-shadow:0 0 2px #fff, 0 0 6px #fff; white-space:nowrap;\">"
            f"{name}</div>"
        )
        folium.Marker(
            location=[pt.y, pt.x],
            icon=folium.DivIcon(html=html, icon_size=(0, 0), icon_anchor=(0, 0)),
            z_index_offset=1000,
        ).add_to(m)

def draw_admin_boundaries(m, layer_name: str):
    """Desenha limites + caso Isócronas em 'limites', aplica cor #836e60 para classes 1/3/4/6."""
    if gpd is None or folium is None:
        return
    if layer_name == "Isócronas (área de influência)":
        gdf_iso = load_isocronas()
        if gdf_iso is None:
            return
        gf = gdf_iso.copy()
        gf["__is_transicao__"] = gf.get("nova_class").isin([1, 3, 4, 6])
        folium.GeoJson(
            data=gf.to_json(),
            name="Isócronas (área de influência)",
            style_function=lambda f: {
                "fillOpacity": 0.0,
                "color": "#836e60" if f["properties"].get("__is_transicao__", False) else "#222222",
                "weight": 1.2,
            },
        ).add_to(m)
        # Rótulos (usa 'nova_class')
        add_centroid_labels(m, gf, "nova_class")
        return

    gdf = load_admin_layer(layer_name)
    if gdf is None:
        return

    # coluna de rótulo
    cols_lower = {c.lower(): c for c in gdf.columns}
    label_col = None
    if "ds_nome" in cols_lower:
        label_col = cols_lower["ds_nome"]
    elif "sp_nome" in cols_lower:
        label_col = cols_lower["sp_nome"]

    # contorno
    gf_line = gdf[["geometry"]].copy()
    gf_line["geometry"] = gf_line.geometry.boundary
    gf_line["geometry"] = gf_line.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    folium.GeoJson(
        data=gf_line.to_json(),
        name=f"{layer_name} (contorno)",
        style_function=lambda f: {"fillOpacity": 0, "color": "#000000", "weight": 1.2},
    ).add_to(m)

    if label_col:
        add_centroid_labels(m, gdf, label_col)

# ---------- variáveis (setores/isócronas) ----------

CLUSTER_COLORS = {
    0: "#bf7db2",
    1: "#f7bd6a",
    2: "#cf651f",
    3: "#ede4e6",
    4: "#793393",
    "outros": "#c8c8c8",
}
CLUSTER_LABELS = {
    1: "Periférico com predominância residencial de alta densidade construtiva",
    2: "Uso misto de média densidade construtiva",
    3: "Periférico com predominância residencial de média densidade construtiva",
    4: "Verticalizado de uso-misto",
    5: "Predominância de uso comercial e serviços",
}

ISO_CLASS_COLORS = {
    0: "#542788",  # Predominância uso misto
    1: "#f7f7f7",  # Zona de transição local
    2: "#d8daeb",  # Periférico res. média densidade
    3: "#b35806",  # Transição central verticalizada
    4: "#b2abd2",  # periférico adensado em transição
    5: "#8073ac",  # Centralidade comercial e serviços
    6: "#fdb863",  # Predominância res. média densidade
    7: "#7f3b08",  # Áreas íngremes e de encosta
    8: "#e08214",  # Alta densidade residencial
    9: "#fee0b6",  # Central verticalizado
}

def _find_key(df_or_gdf, *cands) -> Optional[str]:
    cols = {c.lower(): c for c in df_or_gdf.columns}
    for c in cands:
        if c.lower() in cols:
            return cols[c.lower()]
    return None

def choropleth_setores(
    m,
    setores_gdf: gpd.GeoDataFrame,
    valores_df: pd.DataFrame,
    var_col: str,
    legenda_titulo: str,
):
    """Pinta os setores sem borda (apenas preenchimento)."""
    if branca is None:
        st.warning("Pacote 'branca' não disponível para a legenda.")
        return

    # chave de setor
    key = _find_key(setores_gdf, "CD_GEOCODI", "CD_SETOR")
    key2 = _find_key(valores_df, "CD_GEOCODI", "CD_SETOR")
    if not key or not key2:
        st.info("Coluna de setor não encontrada (CD_GEOCODI/CD_SETOR).")
        return

    gdf = setores_gdf[[key, "geometry"]].rename(columns={key: "setor_key"})
    df = valores_df[[key2, var_col]].rename(columns={key2: "setor_key", var_col: "valor"})
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    mg = gdf.merge(df, on="setor_key", how="inner").dropna(subset=["valor"])
    if mg.empty:
        st.info("Sem dados para desenhar a variável.")
        return

    # paleta contínua (gradiente exigido)
    vmin, vmax = float(mg["valor"].min()), float(mg["valor"].max())
    cmap = branca.colormap.LinearColormap(colors=GRADIENTE_BASE, vmin=vmin, vmax=vmax)
    cmap.caption = legenda_titulo

    def style_fn(feat):
        v = feat["properties"].get("valor")
        return {"fillOpacity": 0.8, "weight": 0, "color": "#00000000", "fillColor": cmap(v) if v is not None else "#cccccc"}

    folium.GeoJson(
        data=mg.to_json(),
        name=legenda_titulo,
        style_function=style_fn,
        tooltip=folium.features.GeoJsonTooltip(
            fields=["setor_key", "valor"],
            aliases=["Setor:", f"{legenda_titulo}: "],
            sticky=True,
        ),
    ).add_to(m)

    cmap.add_to(m)

def choropleth_clusters(m, setores_gdf, valores_df):
    """Clusters categóricos (cores fixas + legenda)."""
    key = _find_key(setores_gdf, "CD_GEOCODI", "CD_SETOR")
    key2 = _find_key(valores_df, "CD_GEOCODI", "CD_SETOR")
    cluster_col = _find_key(valores_df, "cluster")
    if not (key and key2 and cluster_col):
        st.info("Não encontrei coluna 'cluster' com chave de setor.")
        return

    gdf = setores_gdf[[key, "geometry"]].rename(columns={key: "setor_key"})
    df = valores_df[[key2, cluster_col]].rename(columns={key2: "setor_key", cluster_col: "cluster"})
    mg = gdf.merge(df, on="setor_key", how="inner")
    if mg.empty:
        st.info("Sem dados de cluster.")
        return

    def style_fn(feat):
        v = feat["properties"].get("cluster")
        try:
            v_int = int(v)
        except Exception:
            return {"fillOpacity": 0.85, "weight": 0, "color": "#00000000", "fillColor": CLUSTER_COLORS.get("outros")}
        return {"fillOpacity": 0.85, "weight": 0, "color": "#00000000", "fillColor": CLUSTER_COLORS.get(v_int, CLUSTER_COLORS["outros"])}

    folium.GeoJson(
        data=mg.to_json(),
        name="Cluster",
        style_function=style_fn,
        tooltip=folium.features.GeoJsonTooltip(
            fields=["setor_key", "cluster"],
            aliases=["Setor:", "Cluster:"],
            sticky=True,
        ),
    ).add_to(m)

    # legenda discreta
    html = ["<div style='background:white;padding:8px 10px;border:1px solid #ccc;border-radius:8px'>",
            "<b>Cluster (tipologia urbana)</b><br>"]
    for k, color in CLUSTER_COLORS.items():
        if k == "outros":
            continue
        html.append(f"<div style='display:flex;align-items:center;margin:4px 0'>"
                    f"<span style='width:14px;height:14px;background:{color};display:inline-block;margin-right:6px;border:1px solid #888'></span>"
                    f"{k} – {CLUSTER_LABELS.get(k+1, '')}</div>")
    html.append("</div>")
    folium.Marker(
        location=[-90, -180],  # ficará fora do mapa; será movido pela UI do Leaflet
        icon=folium.DivIcon(html="".join(html)),
    ).add_to(m)

def choropleth_isocronas(m, gdf_iso: gpd.GeoDataFrame):
    """Variável 'Área de influência de bairro' (nova_class com paleta exigida)."""
    col = _find_key(gdf_iso, "nova_class")
    if col is None:
        st.info("Coluna 'nova_class' não encontrada nas isócronas.")
        return
    gf = gdf_iso[[col, "geometry"]].rename(columns={col: "classe"})
    gf["classe"] = pd.to_numeric(gf["classe"], errors="coerce").astype("Int64")

    def style_fn(feat):
        v = feat["properties"].get("classe")
        return {"fillOpacity": 0.75, "weight": 0, "color": "#00000000", "fillColor": ISO_CLASS_COLORS.get(int(v) if v is not None else -1, "#cccccc")}

    folium.GeoJson(
        data=gf.to_json(),
        name="Área de influência de bairro",
        style_function=style_fn,
        tooltip=folium.features.GeoJsonTooltip(
            fields=["classe"], aliases=["Classe:"], sticky=True
        ),
    ).add_to(m)

    # legenda discreta
    html = ["<div style='background:white;padding:8px 10px;border:1px solid #ccc;border-radius:8px;max-width:260px'>",
            "<b>Área de influência de bairro</b><br>"]
    iso_labels = {
        0: "Predominância uso misto",
        1: "Zona de transição local",
        2: "Periférico residencial média dens.",
        3: "Transição central verticalizada",
        4: "Periférico adensado em transição",
        5: "Centralidade comercial/serviços",
        6: "Predominância residencial média dens.",
        7: "Áreas íngremes/encosta",
        8: "Alta densidade residencial",
        9: "Central verticalizado",
    }
    for k, color in ISO_CLASS_COLORS.items():
        html.append(f"<div style='display:flex;align-items:center;margin:4px 0'>"
                    f"<span style='width:14px;height:14px;background:{color};display:inline-block;margin-right:6px;border:1px solid #888'></span>"
                    f"{k} – {iso_labels.get(k,'')}</div>")
    html.append("</div>")
    folium.Marker(location=[-90, -180], icon=folium.DivIcon(html="".join(html))).add_to(m)


# ============================================================================
# App
# ============================================================================
def main() -> None:
    inject_css()

    # Cabeçalho
    with st.container():
        c1, c2 = st.columns([1, 7])
        with c1:
            lp = get_logo_path()
            if lp:
                st.image(lp, width=140)
        with c2:
            st.markdown(
                """
                <div class="pb-header">
                    <div style="display:flex;flex-direction:column">
                        <div class="pb-title">PlanBairros</div>
                        <div class="pb-subtitle">Plataforma de visualização e planejamento em escala de bairro</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Abas de topo (placeholders)
    build_tabs()

    # Layout com filtros na esquerda
    left, map_col = st.columns([1, 5], gap="small")

    with left:
        var_key, limite_choice, labels_on = build_left_controls("left_")

    # dados necessários
    gdf_limite = None
    if limite_choice != "Isócronas (área de influência)":
        gdf_limite = load_admin_layer(limite_choice)
    gdf_setores = load_setores()
    df_set_vals = load_setores_values()
    gdf_iso = load_isocronas()

    with map_col:
        if folium is not None and st_folium is not None:
            center = (-23.55, -46.63)
            if gdf_limite is not None and len(gdf_limite) > 0:
                minx, miny, maxx, maxy = gdf_limite.total_bounds
                center = ((miny + maxy) / 2, (minx + maxx) / 2)
            elif gdf_iso is not None and not gdf_iso.empty:
                minx, miny, maxx, maxy = gdf_iso.total_bounds
                center = ((miny + maxy) / 2, (minx + maxx) / 2)

            fmap = make_satellite_map(center=center, zoom=10, tiles_opacity=0.5)
            inject_label_scaler(fmap, min_px=12, max_px=22, min_zoom=9, max_zoom=18)

            # Limites (sem setores). Se for isócronas, usa a regra #836e60 para classes 1/3/4/6
            draw_admin_boundaries(fmap, limite_choice)

            # Variáveis
            if var_key in {"populacao", "densidade_demografica", "diferenca_elevacao", "elevacao"}:
                if gdf_setores is not None and df_set_vals is not None:
                    titulo = {
                        "populacao": "População (Pessoa/hec)",
                        "densidade_demografica": "Densidade demográfica (hab/he)",
                        "diferenca_elevacao": "Variação de elevação média",
                        "elevacao": "Elevação média",
                    }[var_key]
                    if var_key in df_set_vals.columns or var_key in {c.lower() for c in df_set_vals.columns}:
                        # normaliza nome exato
                        real = next((c for c in df_set_vals.columns if c.lower() == var_key.lower()), var_key)
                        choropleth_setores(fmap, gdf_setores, df_set_vals, real, titulo)
                    else:
                        st.info(f"Coluna '{var_key}' não encontrada no arquivo de valores por setor.")

            elif var_key == "cluster":
                if gdf_setores is not None and df_set_vals is not None:
                    choropleth_clusters(fmap, gdf_setores, df_set_vals)

            elif var_key == "isocronas":
                if gdf_iso is not None:
                    choropleth_isocronas(fmap, gdf_iso)
                else:
                    st.info("Isócronas não encontradas para a variável.")

            # Rótulos permanentes extras no limite escolhido (se aplicável) já foram aplicados em draw_admin_boundaries
            st_folium(fmap, use_container_width=True, height=820)
        else:
            st.info("Para o mapa satélite, instale `folium` e `streamlit-folium`.")


if __name__ == "__main__":
    main()
