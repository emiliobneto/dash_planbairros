# -*- coding: utf-8 -*-
"""
PlanBairros – app.py (reescrito com correções de performance e robustez)

Principais ajustes:
• Removido cálculo pesado de centroide via `unary_union`; agora usa `total_bounds` (O(1)).
• `st_folium` só é chamado se `folium` **e** `streamlit-folium` estiverem disponíveis.
• Leituras de Parquet protegidas com `try/except` + mensagens amigáveis.
• Controles limpos e reordenados: Limites Administrativos → Variáveis → Métricas → Informações.
• Limites renderizados apenas como contorno; fundo satélite com 50% de opacidade.
• Variável "Densidade" aparece no mapa (choropleth por setor) e histograma ao lado; "Zoneamento" reservado.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import re

import pandas as pd
import plotly.express as px
import streamlit as st

# Bibliotecas opcionais para dados espaciais e mapa satélite
try:
    import geopandas as gpd  # type: ignore
    import folium  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
except Exception:  # noqa: BLE001
    gpd = None  # type: ignore
    folium = None  # type: ignore
    st_folium = None  # type: ignore

# ============================================================
# Configuração global da página
# ============================================================
st.set_page_config(
    page_title="PlanBairros",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# Paleta da marca
# ============================================================
PB_COLORS = {
    "amarelo": "#F4DD63",
    "verde": "#B1BF7C",
    "laranja": "#D58243",
    "telha": "#C65534",
    "teal": "#6FA097",
    "navy": "#14407D",
}

# ============================================================
# Estilo e assets
# ============================================================

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
            :root {{
                --pb-amarelo: {PB_COLORS['amarelo']};
                --pb-verde:   {PB_COLORS['verde']};
                --pb-laranja: {PB_COLORS['laranja']};
                --pb-telha:   {PB_COLORS['telha']};
                --pb-teal:    {PB_COLORS['teal']};
                --pb-navy:    {PB_COLORS['navy']};
            }}
            html, body, .stApp {{ font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important; }}
            .pb-header {{ background: var(--pb-navy); color: #fff; border-radius: 18px; padding: 20px 24px; min-height: 110px; display: flex; align-items: center; gap: 16px; }}
            .pb-title {{ font-size: 2.2rem; line-height: 1.15; font-weight: 700; letter-spacing: .5px; }}
            .pb-subtitle {{ opacity: .95; margin-top: 4px; font-size: 1.05rem; }}
            .pb-card {{ background:#fff; border:1px solid rgba(20,64,125,.10); box-shadow:0 1px 2px rgba(0,0,0,.04); border-radius:16px; padding:16px; }}
            .stTabs [data-baseweb=\"tab-list\"] button[role=\"tab\"] {{ background:transparent; border-bottom:3px solid transparent; font-weight:600; }}
            .stTabs [data-baseweb=\"tab-list\"] button[aria-selected=\"true\"] {{ border-bottom:3px solid var(--pb-teal) !important; color:var(--pb-navy) !important; }}
            footer, #MainMenu {{ visibility:hidden; }}
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

# ============================================================
# Leitura de dados
# ============================================================
BASE_DIR = Path("dash_planbairros")
ADM_DIR = BASE_DIR / "limites_administrativos"
DENS_DIR = BASE_DIR / "densidade"

@st.cache_data(show_spinner=False)
def load_admin_layer(layer_name: str):
    """Lê um layer administrativo (Parquet) e retorna GeoDataFrame WGS84.
    Mostra mensagens amigáveis em caso de falha e não quebra o app.
    """
    if gpd is None:
        st.info("Geopandas não disponível — instale `geopandas`, `pyarrow` e `shapely`.")
        return None
    candidates = [
        ADM_DIR / f"{layer_name}.parquet",
        ADM_DIR / layer_name / f"{layer_name}.parquet",
    ]
    for path in candidates:
        if path.exists():
            try:
                gdf = gpd.read_parquet(path)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Falha ao ler {path.name}: {exc}")
                return None
            try:
                if gdf.crs is None:
                    gdf.set_crs(4326, inplace=True)
                else:
                    gdf = gdf.to_crs(4326)
            except Exception:
                pass
            return gdf
    st.warning(f"Arquivo Parquet não encontrado para '{layer_name}' em {ADM_DIR}.")
    return None

@st.cache_data(show_spinner=False)
def load_density() -> Optional[pd.DataFrame]:
    """Carrega Parquet de densidade/zoneamento."""
    candidates = [
        DENS_DIR.with_suffix(".parquet"),
        DENS_DIR / "densidade.parquet",
        DENS_DIR,
    ]
    for c in candidates:
        try:
            if c.is_file():
                return pd.read_parquet(c)
            if c.is_dir():
                files = list(c.glob("*.parquet"))
                if files:
                    return pd.read_parquet(files[0])
        except Exception as exc:
            st.warning(f"Falha ao ler {c}: {exc}")
            return None
    st.info("Dados de densidade não encontrados em 'dash_planbairros/densidade'.")
    return None

# util: tenta inferir chave de junção do setor censitário

def infer_setor_key(gdf_cols, df_cols) -> Optional[str]:
    patterns = [r"^cd[_-]?setor$", r"^setor[_-]?id$", r"^id[_-]?setor$", r"^setor$", r"^cd_setor_2023$"]
    g = {c.lower() for c in gdf_cols}
    d = {c.lower() for c in df_cols}
    for pat in patterns:
        for col in g:
            if re.match(pat, col) and col in d:
                return col
    common = g.intersection(d)
    return next(iter(common)) if common else None

# ============================================================
# UI Components
# ============================================================

def build_header(logo_path: Optional[str]) -> None:
    with st.container():
        col1, col2 = st.columns([1, 7])
        with col1:
            if logo_path:
                st.image(logo_path, width=140)
            else:
                st.write("")
        with col2:
            st.markdown(
                """
                <div class=\"pb-header\">
                    <div style=\"display:flex;flex-direction:column\">
                        <div class=\"pb-title\">PlanBairros</div>
                        <div class=\"pb-subtitle\">Plataforma de visualização e planejamento em escala de bairro</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

def build_tabs() -> None:
    aba1, aba2, aba3, aba4 = st.tabs(["Aba 1", "Aba 2", "Aba 3", "Aba 4"])
    for i, aba in enumerate([aba1, aba2, aba3, aba4], start=1):
        with aba:
            st.markdown(f"**Conteúdo da Aba {i}** — espaço para textos/explicações.")
    st.write("")


def build_controls() -> Tuple[str, str, str, str]:
    st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
    st.markdown("<h4>Configurações</h4>", unsafe_allow_html=True)

    limite = st.selectbox(
        "Limites Administrativos",
        ["Distritos", "SetoresCensitarios2023", "ZonasOD2023", "Subprefeitura"],
        index=0,
        help="Escolha a desagregação espacial para exibir como contorno.",
        key="pb_limite",
    )

    variavel = st.selectbox(
        "Variáveis",
        ["Densidade", "Zoneamento"],
        index=0,
        help="Variáveis agregadas por Setor Censitário a partir do mapa de densidade.",
        key="pb_variavel",
    )

    metrica = st.selectbox("Métricas", ["—"], index=0, help="Reservado para cálculos futuros.", key="pb_metrica")
    info = st.selectbox("Informações", ["—"], index=0, help="Notas/documentação.", key="pb_info")

    st.markdown("</div>", unsafe_allow_html=True)
    return limite, variavel, metrica, info

# ============================================================
# Visualizações (Folium)
# ============================================================

def make_satellite_map(center=(-23.55, -46.63), zoom=10, tiles_opacity=0.5):
    """Mapa Folium com camada satélite (Google; fallback Esri)."""
    if folium is None:
        st.error("Bibliotecas de mapa (folium/streamlit-folium) não disponíveis.")
        return None
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    try:
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google",
            name="Google Satellite",
            overlay=True,
            control=False,
            opacity=tiles_opacity,
        ).add_to(m)
    except Exception:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Esri Satellite",
            overlay=True,
            control=False,
            opacity=tiles_opacity,
        ).add_to(m)
    return m


def add_admin_outline(m, gdf, color=PB_COLORS["navy"], weight=2):
    if gdf is None or folium is None:
        return
    try:
        folium.GeoJson(
            gdf.to_json(),
            name="Limites",
            style_function=lambda feat: {"fillOpacity": 0.0, "color": color, "weight": weight},
        ).add_to(m)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Não foi possível desenhar os limites: {exc}")


def plot_density_variable(m, setores_gdf, dens_df, var_name="densidade"):
    """Choropleth de densidade por Setor Censitário."""
    if folium is None or gpd is None or setores_gdf is None or dens_df is None:
        return
    key = infer_setor_key(setores_gdf.columns, dens_df.columns)
    if key is None:
        st.info("Não foi possível inferir a chave de junção entre setores e densidade.")
        return
    gdf = setores_gdf.rename(columns={key: "setor_key"})
    df = dens_df.rename(columns={key: "setor_key"})

    if var_name.lower().startswith("dens"):
        if not any(c.lower() == "densidade" for c in df.columns):
            cand = [c for c in df.columns if re.search(r"dens", c, re.I)]
            if cand:
                df = df.rename(columns={cand[0]: "densidade"})
        df = df[["setor_key", "densidade"]].dropna()
        merged = gdf.merge(df, on="setor_key", how="inner")
        try:
            q = merged["densidade"].quantile([0, .2, .4, .6, .8, 1]).tolist()
            palette = ["#F4DD63", "#B1BF7C", "#6FA097", "#D58243", "#14407D"]
            def style_fn(feat):
                v = feat["properties"].get("densidade")
                idx = sum(v >= x for x in q[:-1]) if v is not None else 0
                return {"fillOpacity": 0.65, "weight": 0.6, "color": "#ffffff", "fillColor": palette[min(idx, len(palette)-1)]}
            folium.GeoJson(
                merged.to_json(),
                name="Densidade",
                style_function=style_fn,
                tooltip=folium.features.GeoJsonTooltip(fields=["densidade"], aliases=["Densidade:"]),
            ).add_to(m)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Falha ao desenhar densidade: {exc}")


def bar_for_density(dens_df: Optional[pd.DataFrame]):
    if dens_df is None:
        st.info("Sem dados de densidade para o gráfico.")
        return
    # tenta localizar a coluna de densidade
    import re as _re
    col = next((c for c in dens_df.columns if c.lower() == "densidade" or _re.search(r"dens", c, _re.I)), None)
    if col is None:
        st.info("Coluna de densidade não encontrada no Parquet.")
        return
    st.subheader("Densidade — distribuição")
    fig = px.histogram(dens_df, x=col, nbins=30, height=540)
    fig.update_traces(marker_color=PB_COLORS["laranja"])  # cor da marca
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Utilidades
# ============================================================

def center_from_gdf_bounds(gdf) -> tuple[float, float]:
    """Centro (lat, lon) a partir do bounding box – rápido e leve."""
    minx, miny, maxx, maxy = gdf.total_bounds
    lat = (miny + maxy) / 2
    lon = (minx + maxx) / 2
    return (lat, lon)

# ============================================================
# App
# ============================================================

def main() -> None:
    inject_css()
    logo_path = get_logo_path()
    build_header(logo_path)
    st.write("")
    build_tabs()

    try:
        left, center, right = st.columns([1, 2, 2], gap="large")
    except TypeError:
        left, center, right = st.columns([1, 2, 2])

    with left:
        limite, variavel, _, _ = build_controls()

    # Carregamentos
    gdf_limite = load_admin_layer(limite)
    dens_df = load_density() if variavel in ("Densidade", "Zoneamento") else None

    # Centro/direita — mapa e gráfico
    with center:
        if folium is not None and st_folium is not None:
            center_latlon = (-23.55, -46.63)
            if gdf_limite is not None and len(gdf_limite) > 0:
                center_latlon = center_from_gdf_bounds(gdf_limite)
            fmap = make_satellite_map(center=center_latlon, zoom=10, tiles_opacity=0.5)
            if fmap is not None:
                if gdf_limite is not None:
                    add_admin_outline(fmap, gdf_limite)
                if variavel == "Densidade" and dens_df is not None:
                    setores = load_admin_layer("SetoresCensitarios2023")
                    if setores is not None:
                        plot_density_variable(fmap, setores, dens_df, var_name="densidade")
                st_folium(fmap, use_container_width=True, height=600)
        else:
            st.info("Para o mapa satélite, instale `folium` e `streamlit-folium`.")

    with right:
        if variavel == "Densidade":
            bar_for_density(dens_df)
        else:
            st.subheader("Visualização")
            st.write("Selecione **Densidade** para ver o gráfico. 'Zoneamento' será integrado em breve.")


if __name__ == "__main__":
    main()
