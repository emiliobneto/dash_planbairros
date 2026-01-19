from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import re

import numpy as np
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
from unicodedata import normalize as _ud_norm

BASE_DIR = Path("dash_planbairros")
ADM_DIR = BASE_DIR / "limites_administrativos"
DENS_DIR = BASE_DIR / "densidade"


def _slug(s: str) -> str:
    s2 = _ud_norm("NFKD", str(s)).encode("ASCII", "ignore").decode("ASCII")
    return re.sub(r"[^a-z0-9]+", "", s2.strip().lower())


def _first_parquet_matching(folder: Path, name_candidates: list[str]) -> Optional[Path]:
    """Procura por .parquet em `folder` cujo nome (sem extensão) bata com qualquer candidato normalizado.
    Retorna o primeiro Path encontrado; None caso contrário.
    """
    if not folder.exists():
        return None
    wanted = {_slug(n) for n in name_candidates}
    for fp in folder.glob("*.parquet"):
        if _slug(fp.stem) in wanted:
            return fp
    return None

# Mapas administrativos: mapeia rótulos do dropdown para nomes reais de arquivo
ADMIN_NAME_MAP = {
    "Distritos": ["Distritos"],
    "SetoresCensitarios2023": ["SetoresCensitarios2023", "SetoresCensitarios"],
    "ZonasOD2023": ["ZonasOD2023", "ZonasOD"],
    "Subprefeitura": ["Subprefeitura", "Subprefeituras", "subprefeitura"],
}

@st.cache_data(show_spinner=False)
def load_admin_layer(layer_name: str):
    if gpd is None:
        st.info("Geopandas não disponível — instale `geopandas`, `pyarrow` e `shapely`.")
        return None
    # tenta nomes exatos e variações observadas no repositório
    cand_names = ADMIN_NAME_MAP.get(layer_name, [layer_name])
    path = _first_parquet_matching(ADM_DIR, cand_names)
    if path is None:
        st.warning(f"Arquivo Parquet não encontrado para '{layer_name}' em {ADM_DIR}.")
        return None
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

# Variáveis (densidade): nomes observados no repo
DENSITY_CANDIDATES = [
    "Densidade", "Densidade2023",  # nomes corretos
    "Desindade", "Desindade2023",   # variação com typo vista no repo
]

@st.cache_data(show_spinner=False)
def load_density() -> Optional[pd.DataFrame]:
    # aceita arquivo direto em `dash_planbairros/densidade.parquet` OU dentro da pasta `densidade/`
    direct = DENS_DIR.with_suffix(".parquet")
    if direct.exists():
        try:
            return pd.read_parquet(direct)
        except Exception as exc:
            st.warning(f"Falha ao ler {direct}: {exc}")
            return None
    # procura candidatos por nome
    p = _first_parquet_matching(DENS_DIR, DENSITY_CANDIDATES)
    if p is not None:
        try:
            return pd.read_parquet(p)
        except Exception as exc:
            st.warning(f"Falha ao ler {p}: {exc}")
            return None
    # fallback: primeiro parquet encontrado
    if DENS_DIR.exists():
        files = list(DENS_DIR.glob("*.parquet"))
        if files:
            try:
                return pd.read_parquet(files[0])
            except Exception as exc:
                st.warning(f"Falha ao ler {files[0]}: {exc}")
                return None
    st.info("Dados de densidade não encontrados em 'dash_planbairros/densidade'.")
    return None

# ============================================================
# Utilidades
# ============================================================

def center_from_gdf_bounds(gdf) -> tuple[float, float]:
    minx, miny, maxx, maxy = gdf.total_bounds
    lat = (miny + maxy) / 2
    lon = (minx + maxx) / 2
    return (lat, lon)

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
# Visualizações (Folium) com guardas de performance
# ============================================================

CHORO_MAX_FEATURES = 8000   # acima disso, cai para marcadores de ponto
POINTS_SAMPLE_MAX  = 2000   # limite de amostras de pontos
SIMPLIFY_TOL       = 0.0005 # ~55m em latitude


def make_satellite_map(center=(-23.55, -46.63), zoom=10, tiles_opacity=0.5):
    """Mapa Folium com camada satélite (Google; fallback Esri) e pane exclusivo para vetores."""
    if folium is None:
        st.error("Bibliotecas de mapa (folium/streamlit-folium) não disponíveis.")
        return None
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    # pane para garantir que vetores fiquem acima do tile
    try:
        folium.map.CustomPane("vectors", z_index=650).add_to(m)
    except Exception:
        pass
    try:
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google",
            name="Google Satellite",
            overlay=False,  # base layer
            control=False,
            opacity=tiles_opacity,
        ).add_to(m)
    except Exception:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Esri Satellite",
            overlay=False,
            control=False,
            opacity=tiles_opacity,
        ).add_to(m)
    return m


def add_admin_outline(m, gdf, color="#000000", weight=1.0):
    """Desenha somente o contorno dos polígonos em um pane acima do tile.
    - Converte polígonos em *linhas* (boundary) para ficar mais leve e visível.
    - Simplifica suavemente para reduzir payload.
    """
    if gdf is None or folium is None:
        return
    try:
        gdf_lines = gdf[["geometry"]].copy()
        # só as linhas do contorno
        gdf_lines["geometry"] = gdf_lines.geometry.boundary
        # simplificação leve
        gdf_lines["geometry"] = gdf_lines.geometry.simplify(0.0005, preserve_topology=True)
        folium.GeoJson(
            data=gdf_lines.__geo_interface__,
            name="Limites",
            pane="vectors",
            style_function=lambda feat: {
                "fillOpacity": 0.0,
                "color": color,
                "weight": weight,
            },
        ).add_to(m)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Não foi possível desenhar os limites: {exc}")
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
    """Choropleth de densidade por Setor Censitário.
    Garante pane dos vetores e estilização acima do tile.
    """
    if folium is None or gpd is None or setores_gdf is None or dens_df is None:
        return
    key = infer_setor_key(setores_gdf.columns, dens_df.columns)
    if key is None:
        st.info("Não foi possível inferir a chave de junção entre setores e densidade.")
        return
    gdf = setores_gdf.rename(columns={key: "setor_key"})[["setor_key", "geometry"]].copy()
    df = dens_df.rename(columns={key: "setor_key"}).copy()
    # detecta coluna de densidade
    dens_col = next((c for c in df.columns if c.lower() == "densidade" or re.search(r"dens", c, re.I)), None)
    if dens_col is None:
        st.info("Coluna de densidade não encontrada no Parquet.")
        return
    df = df[["setor_key", dens_col]].rename(columns={dens_col: "densidade"}).dropna()
    merged = gdf.merge(df, on="setor_key", how="inner")
    if len(merged) == 0:
        st.info("Sem setores para desenhar.")
        return
    # simplificação leve
    mg = merged.copy()
    mg["geometry"] = mg.geometry.simplify(0.0005, preserve_topology=True)
    q = mg["densidade"].quantile([0, .2, .4, .6, .8, 1]).tolist()
    palette = ["#F4DD63", "#B1BF7C", "#6FA097", "#D58243", "#14407D"]
    def style_fn(feat):
        v = feat["properties"].get("densidade")
        idx = sum(v >= x for x in q[:-1]) if v is not None else 0
        return {"fillOpacity": 0.65, "weight": 0.6, "color": "#ffffff", "fillColor": palette[min(idx, len(palette)-1)]}
    try:
        folium.GeoJson(
            data=mg.__geo_interface__,
            name="Densidade",
            pane="vectors",
            style_function=style_fn,
            tooltip=folium.features.GeoJsonTooltip(fields=["densidade"], aliases=["Densidade:"]),
        ).add_to(m)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Falha ao desenhar densidade: {exc}")
    if folium is None or gpd is None or setores_gdf is None or dens_df is None:
        return

    key = infer_setor_key(setores_gdf.columns, dens_df.columns)
    if key is None:
        st.info("Não foi possível inferir a chave de junção entre setores e densidade.")
        return

    gdf = setores_gdf.rename(columns={key: "setor_key"})[["setor_key", "geometry"]].copy()
    df = dens_df.rename(columns={key: "setor_key"}).copy()

    # detecta coluna de densidade
    dens_col = next((c for c in df.columns if c.lower() == "densidade" or re.search(r"dens", c, re.I)), None)
    if dens_col is None:
        st.info("Coluna de densidade não encontrada no Parquet.")
        return

    df = df[["setor_key", dens_col]].rename(columns={dens_col: "densidade"}).dropna()
    merged = gdf.merge(df, on="setor_key", how="inner")

    # Escolha do modo de desenho conforme tamanho
    nfeat = len(merged)
    if nfeat == 0:
        st.info("Sem setores para desenhar.")
        return

    if nfeat <= CHORO_MAX_FEATURES:
        # Simplificação leve para reduzir payload
        try:
            mg = merged.copy()
            mg["geometry"] = mg.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
            q = mg["densidade"].quantile([0, .2, .4, .6, .8, 1]).tolist()
            palette = ["#F4DD63", "#B1BF7C", "#6FA097", "#D58243", "#14407D"]
            def style_fn(feat):
                v = feat["properties"].get("densidade")
                idx = sum(v >= x for x in q[:-1]) if v is not None else 0
                return {"fillOpacity": 0.65, "weight": 0.6, "color": "#ffffff", "fillColor": palette[min(idx, len(palette)-1)]}
            folium.GeoJson(
                mg.to_json(),
                name="Densidade",
                style_function=style_fn,
                tooltip=folium.features.GeoJsonTooltip(fields=["densidade"], aliases=["Densidade:"]),
            ).add_to(m)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Falha ao desenhar densidade (choropleth): {exc}")
    else:
        # Modo pontos: mais leve para muitos polígonos
        try:
            rp = merged.copy()
            # ponto representativo dentro do polígono
            rp["pt"] = rp.geometry.representative_point()
            rp["lat"] = rp["pt"].y
            rp["lon"] = rp["pt"].x
            rp = rp[["lat", "lon", "densidade"]]
            # amostra para não sobrecarregar o front
            if len(rp) > POINTS_SAMPLE_MAX:
                rp = rp.sample(POINTS_SAMPLE_MAX, random_state=42)
            # escala simples de raio por quantis
            q = rp["densidade"].quantile([.2, .4, .6, .8]).tolist()
            for _, r in rp.iterrows():
                v = r["densidade"]
                radius = 3 + 2 * sum(v >= x for x in q)
                folium.CircleMarker(
                    location=(r["lat"], r["lon"]),
                    radius=radius,
                    color="#ffffff",
                    weight=0.6,
                    fill=True,
                    fill_color=PB_COLORS["laranja"],
                    fill_opacity=0.7,
                    tooltip=f"Densidade: {v}",
                ).add_to(m)
        except Exception as exc:
            st.warning(f"Falha ao desenhar densidade (pontos): {exc}")


def bar_for_density(dens_df: Optional[pd.DataFrame]):
    if dens_df is None:
        st.info("Sem dados de densidade para o gráfico.")
        return
    col = next((c for c in dens_df.columns if c.lower() == "densidade" or re.search(r"dens", c, re.I)), None)
    if col is None:
        st.info("Coluna de densidade não encontrada no Parquet.")
        return
    st.subheader("Densidade — distribuição")
    # Remove outliers extremos para visual mais estável
    vals = dens_df[col].astype(float)
    q1, q99 = np.nanpercentile(vals, 1), np.nanpercentile(vals, 99)
    vals_clip = np.clip(vals, q1, q99)
    fig = px.histogram(vals_clip, nbins=30, height=540)
    fig.update_traces(marker_color=PB_COLORS["laranja"])  # cor da marca
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# App
# ============================================================

def main() -> None:
    inject_css()
    logo_path = get_logo_path()
    with st.spinner("Carregando interface..."):
        # Header e abas
        with st.container():
            colh1, colh2 = st.columns([1, 7])
            with colh1:
                if logo_path:
                    st.image(logo_path, width=140)
                else:
                    st.write("")
            with colh2:
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
        st.write("")
        aba1, aba2, aba3, aba4 = st.tabs(["Aba 1", "Aba 2", "Aba 3", "Aba 4"])
        for i, aba in enumerate([aba1, aba2, aba3, aba4], start=1):
            with aba:
                st.markdown(f"**Conteúdo da Aba {i}** — espaço para textos/explicações.")
        st.write("")

    try:
        left, center, right = st.columns([1, 2, 2], gap="large")
    except TypeError:
        left, center, right = st.columns([1, 2, 2])

    with left:
        limite, variavel, _, _ = build_controls()

    # Carregamentos (com spinner leve)
    with st.spinner("Lendo camadas..."):
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


