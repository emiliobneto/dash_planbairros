from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
from unicodedata import normalize as _ud_norm
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
# Leitura de dados (robusta a nomes)
# ============================================================
BASE_DIR = Path("dash_planbairros")
ADM_DIR = BASE_DIR / "limites_administrativos"
DENS_DIR = BASE_DIR / "densidade"


def _slug(s: str) -> str:
    s2 = _ud_norm("NFKD", str(s)).encode("ASCII", "ignore").decode("ASCII")
    return re.sub(r"[^a-z0-9]+", "", s2.strip().lower())


def _first_parquet_matching(folder: Path, name_candidates: list[str]) -> Optional[Path]:
    if not folder.exists():
        return None
    wanted = {_slug(n) for n in name_candidates}
    for fp in folder.glob("*.parquet"):
        if _slug(fp.stem) in wanted:
            return fp
    return None

# nomes reais observados
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
    path = _first_parquet_matching(ADM_DIR, ADMIN_NAME_MAP.get(layer_name, [layer_name]))
    if path is None:
        st.warning(f"Arquivo Parquet não encontrado para '{layer_name}' em {ADM_DIR}.")
        return None
    # leitura direta
    try:
        gdf = gpd.read_parquet(path)
    except Exception:
        # fallback: pandas + reconstrução de geometria
        try:
            pdf = pd.read_parquet(path)
        except Exception as exc:
            st.warning(f"Falha ao ler {path.name}: {exc}")
            return None
        geom_col = next((c for c in pdf.columns if c.lower() in ("geometry", "geom", "wkb", "wkt")), None)
        if geom_col is None:
            st.warning(f"{path.name}: coluna de geometria não encontrada.")
            return None
        try:
            from shapely import wkb, wkt
            vals = pdf[geom_col]
            if vals.dropna().astype(str).str.startswith("POLY").any():
                geo = vals.dropna().apply(wkt.loads)
            else:
                geo = vals.dropna().apply(lambda b: wkb.loads(b, hex=isinstance(b, str)))
            gdf = gpd.GeoDataFrame(pdf.drop(columns=[geom_col]), geometry=geo)
        except Exception as exc:
            st.warning(f"{path.name}: não foi possível reconstruir geometria ({exc}).")
            return None
    # CRS → WGS84
    try:
        if gdf.crs is None:
            gdf.set_crs(4326, inplace=True)
        else:
            gdf = gdf.to_crs(4326)
    except Exception:
        pass
    return gdf

# densidade: aceita possíveis nomes/typos
DENSITY_CANDIDATES = ["Densidade", "Densidade2023", "Desindade", "Desindade2023", "HabitacaoPrecaria"]

@st.cache_data(show_spinner=False)
def load_density() -> Optional[pd.DataFrame]:
    direct = DENS_DIR.with_suffix(".parquet")
    if direct.exists():
        try:
            return pd.read_parquet(direct)
        except Exception as exc:
            st.warning(f"Falha ao ler {direct}: {exc}")
            return None
    p = _first_parquet_matching(DENS_DIR, DENSITY_CANDIDATES)
    if p is not None:
        try:
            return pd.read_parquet(p)
        except Exception as exc:
            st.warning(f"Falha ao ler {p}: {exc}")
            return None
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

def infer_setor_key(gdf_cols, df_cols) -> Optional[str]:
    """Coluna de setor com prioridade para CD_GEOCODI / CD_SETOR (case-insensitive)."""
    targets = ["cd_geocodi", "cd_setor", "cd_setor_2023", "setor", "setor_id"]
    g = {c.lower(): c for c in gdf_cols}
    d = {c.lower(): c for c in df_cols}
    for t in targets:
        if t in g and t in d:
            return g[t]
    inter = set(g.keys()).intersection(d.keys())
    return g[next(iter(inter))] if inter else None


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
# Visualizações (Folium)
# ============================================================

CHORO_MAX_FEATURES = 8000
SIMPLIFY_TOL = 0.0005  # ~55m em latitude


def make_satellite_map(center=(-23.55, -46.63), zoom=10, tiles_opacity=0.5):
    if folium is None:
        st.error("Bibliotecas de mapa (folium/streamlit-folium) não disponíveis.")
        return None
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    try:
        folium.map.CustomPane("vectors", z_index=650).add_to(m)
    except Exception:
        pass
    try:
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google",
            name="Google Satellite",
            overlay=False,
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


def add_admin_outline(m, gdf, layer_name: str, color="#000000", weight=1.0):
    """Desenha contorno (boundary) do layer e mostra tooltip com o nome quando existir."""
    if gdf is None or folium is None:
        return
    try:
        gf = gdf.copy()
        gf["geometry"] = gf.geometry.boundary
        gf["geometry"] = gf.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    except Exception as exc:
        st.warning(f"Falha ao preparar contorno: {exc}")
        return
    # tooltip fields por layer
    fields, aliases = [], []
    cols_lower = {c.lower(): c for c in gdf.columns}
    if layer_name.lower().startswith("distr") and "ds_nome" in cols_lower:
        fields, aliases = [cols_lower["ds_nome"]], ["Distrito:"]
    elif layer_name.lower().startswith("subpref") and "sp_nome" in cols_lower:
        fields, aliases = [cols_lower["sp_nome"]], ["Subprefeitura:"]
    elif "setor" in layer_name.lower():
        # mostra código do setor
        key = infer_setor_key(gdf.columns, gdf.columns) or "CD_SETOR"
        col = next((c for c in gdf.columns if c.lower() == key.lower()), None)
        if col:
            fields, aliases = [col], ["Setor:"]
    try:
        folium.GeoJson(
            data=gf.__geo_interface__,
            name=f"{layer_name} (contorno)",
            pane="vectors",
            style_function=lambda feat: {"fillOpacity": 0.0, "color": color, "weight": weight},
            tooltip=folium.features.GeoJsonTooltip(fields=fields, aliases=aliases) if fields else None,
        ).add_to(m)
    except Exception as exc:
        st.warning(f"Não foi possível desenhar os limites: {exc}")


def plot_density_variable(m, setores_gdf, dens_df):
    """Choropleth de `densidade_hec` por Setor Censitário (inteiro)."""
    if folium is None or gpd is None or setores_gdf is None or dens_df is None:
        return
    # chave de setor
    key_pref = None
    for cand in ("CD_GEOCODI", "CD_SETOR"):
        if cand.lower() in [c.lower() for c in setores_gdf.columns] and cand.lower() in [c.lower() for c in dens_df.columns]:
            key_pref = cand
            break
    key = key_pref or infer_setor_key(setores_gdf.columns, dens_df.columns)
    if key is None:
        st.info("Não foi possível identificar a coluna de setor (CD_GEOCODI/CD_SETOR).")
        return
    # nomes reais
    def _real(df, name):
        return next((c for c in df.columns if c.lower()==name.lower()), name)
    key_g = _real(setores_gdf, key)
    key_d = _real(dens_df, key)
    val_d = _real(dens_df, "densidade_hec")
    if val_d not in dens_df.columns:
        st.info("Coluna `densidade_hec` não encontrada no Parquet de densidade.")
        return
    gdf = setores_gdf[[key_g, "geometry"]].copy().rename(columns={key_g:"setor_key"})
    df = dens_df[[key_d, val_d]].copy().rename(columns={key_d:"setor_key", val_d:"densidade_hec"})
    # inteiros para exibição
    df["densidade_hec"] = pd.to_numeric(df["densidade_hec"], errors="coerce").round(0).astype("Int64")
    mg = gdf.merge(df, on="setor_key", how="inner").dropna(subset=["densidade_hec"])
    if len(mg) == 0:
        st.info("Nenhum setor para desenhar após o merge.")
        return
    try:
        mg["geometry"] = mg.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    except Exception:
        pass
    vals = mg["densidade_hec"].astype(float)
    q = vals.quantile([0, .2, .4, .6, .8, 1]).tolist()
    palette = ["#F4DD63", "#B1BF7C", "#6FA097", "#D58243", "#14407D"]
    def style_fn(feat):
        v = feat["properties"].get("densidade_hec")
        idx = sum(float(v) >= x for x in q[:-1]) if v is not None else 0
        return {"fillOpacity": 0.65, "weight": 0.6, "color": "#ffffff", "fillColor": palette[min(idx, len(palette)-1)]}
    try:
        folium.GeoJson(
            data=mg.__geo_interface__,
            name="Densidade",
            pane="vectors",
            style_function=style_fn,
            tooltip=folium.features.GeoJsonTooltip(fields=["setor_key", "densidade_hec"], aliases=["Setor:", "Densidade (hab/ha):"]),
        ).add_to(m)
    except Exception as exc:
        st.warning(f"Falha ao desenhar densidade: {exc}")


def bar_for_density(dens_df: Optional[pd.DataFrame]):
    if dens_df is None or len(dens_df)==0:
        st.info("Sem dados de densidade para o gráfico.")
        return
    col = next((c for c in dens_df.columns if c.lower()=="densidade_hec"), None)
    if col is None:
        st.info("Coluna `densidade_hec` não encontrada no Parquet.")
        return
    vals = pd.to_numeric(dens_df[col], errors="coerce").dropna().round(0).astype(int)
    st.subheader("Densidade (hab/ha) — distribuição")
    fig = px.histogram(vals, nbins=30, height=540)
    fig.update_traces(marker_color=PB_COLORS["laranja"])  # cor da marca
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# App
# ============================================================

def main() -> None:
    inject_css()
    logo_path = get_logo_path()
    # Header
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
    st.write("")
    # Abas
    build_tabs()

    try:
        left, center, right = st.columns([1, 2, 2], gap="large")
    except TypeError:
        left, center, right = st.columns([1, 2, 2])

    with left:
        limite, variavel, _, _ = build_controls()

    # Dados
    gdf_limite = load_admin_layer(limite)
    dens_df = load_density() if variavel == "Densidade" else None

    # Visualizações
    with center:
        if folium is not None and st_folium is not None:
            center_latlon = (-23.55, -46.63)
            if gdf_limite is not None and len(gdf_limite) > 0:
                center_latlon = center_from_gdf_bounds(gdf_limite)
            fmap = make_satellite_map(center=center_latlon, zoom=10, tiles_opacity=0.5)
            if fmap is not None:
                if gdf_limite is not None:
                    add_admin_outline(fmap, gdf_limite, layer_name=limite, color="#000000", weight=1.0)
                if variavel == "Densidade" and dens_df is not None:
                    setores = load_admin_layer("SetoresCensitarios2023")
                    if setores is not None:
                        plot_density_variable(fmap, setores, dens_df)
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
