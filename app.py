# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
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
except Exception:
    gpd = None  # type: ignore
    folium = None  # type: ignore
    st_folium = None  # type: ignore


# ============================================================================
# Configuração da página e paleta
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
            .stTabs [data-baseweb="tab-list"] button[role="tab"] {{ background:transparent; border-bottom:3px solid transparent; font-weight:600; }}
            .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{ border-bottom:3px solid var(--pb-teal) !important; color:var(--pb-navy) !important; }}
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


# ============================================================================
# Resolução robusta de caminhos de dados
# ============================================================================
REPO_ROOT = Path(__file__).resolve().parent


def _resolve_dir(subdir: str) -> Path:
    """
    Retorna o primeiro caminho existente para `subdir` considerando:
    - <repo_root>/<subdir>
    - <repo_root>/dash_planbairros/<subdir>
    """
    for base in (REPO_ROOT, REPO_ROOT / "dash_planbairros"):
        p = base / subdir
        if p.exists():
            return p
    return REPO_ROOT / subdir  # fallback


ADM_DIR = _resolve_dir("limites_administrativos")
DENS_DIR = _resolve_dir("densidade")


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


ADMIN_NAME_MAP = {
    "Distritos": ["Distritos"],
    "SetoresCensitarios2023": ["SetoresCensitarios2023", "SetoresCensitarios"],
    "ZonasOD2023": ["ZonasOD2023", "ZonasOD"],
    "Subprefeitura": ["Subprefeitura", "Subprefeituras", "subprefeitura"],
}


# ============================================================================
# Leitura de dados
# ============================================================================
def _safe_read_parquet(path: Path) -> Optional[pd.DataFrame]:
    """Ignora arquivos muito pequenos/corrompidos; tenta ler com pandas."""
    try:
        if path.stat().st_size < 1024:  # 1 KB — típico de placeholder/LFS
            return None
    except Exception:
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_admin_layer(layer_name: str):
    """Lê um Parquet geográfico e retorna GeoDataFrame em WGS84."""
    if gpd is None:
        st.info("Geopandas não disponível — instale `geopandas`, `shapely` e `pyarrow`.")
        return None

    path = _first_parquet_matching(ADM_DIR, ADMIN_NAME_MAP.get(layer_name, [layer_name]))
    if path is None:
        st.warning(f"Arquivo Parquet não encontrado para '{layer_name}' em {ADM_DIR}.")
        return None

    # tenta ler com geopandas; se falhar reconstrói geometria
    try:
        gdf = gpd.read_parquet(path)
    except Exception:
        pdf = _safe_read_parquet(path)
        if pdf is None:
            st.warning(f"Falha ao ler {path.name}.")
            return None
        geom_col = next((c for c in pdf.columns if c.lower() in ("geometry", "geom", "wkb", "wkt")), None)
        if geom_col is None:
            st.warning(f"{path.name}: coluna de geometria não encontrada.")
            return None
        try:
            from shapely import wkb, wkt  # import local
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
        gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)
    except Exception:
        pass
    return gdf


# densidade (aceita nomes/typos e busca nas duas pastas)
DENSITY_CANDIDATES = ["Densidade", "Densidade2023", "Desindade", "Desindade2023", "HabitacaoPrecaria"]


@st.cache_data(show_spinner=False)
def load_density() -> Optional[pd.DataFrame]:
    """Procura um Parquet válido que contenha a coluna `densidade_hec`."""
    search_order: list[Path] = []

    # 1) <repo>/densidade.parquet
    direct = DENS_DIR.with_suffix(".parquet")
    if direct.exists():
        search_order.append(direct)

    # 2) candidatos por nome em /densidade e em /limites_administrativos
    for base in (DENS_DIR, ADM_DIR):
        if base.exists():
            by_name = _first_parquet_matching(base, DENSITY_CANDIDATES)
            if by_name is not None:
                search_order.append(by_name)
            # varre todos como último recurso
            search_order += sorted(base.glob("*.parquet"))

    # tenta todos, ignorando duplicados, até achar um com densidade_hec
    seen = set()
    for p in search_order:
        if p in seen:
            continue
        seen.add(p)
        df = _safe_read_parquet(p)
        if df is None:
            continue
        if any(c.lower() == "densidade_hec" for c in df.columns):
            return df

    st.info("Dados de densidade não encontrados com coluna `densidade_hec` (substitua eventuais arquivos corrompidos).")
    return None


# ============================================================================
# Utilidades
# ============================================================================
def infer_setor_key(gdf_cols, df_cols) -> Optional[str]:
    """Prioriza CD_GEOCODI / CD_SETOR; fallback para intersecção."""
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
    return ((miny + maxy) / 2, (minx + maxx) / 2)


# ============================================================================
# UI
# ============================================================================
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
                <div class="pb-header">
                    <div style="display:flex;flex-direction:column">
                        <div class="pb-title">PlanBairros</div>
                        <div class="pb-subtitle">Plataforma de visualização e planejamento em escala de bairro</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def build_tabs() -> None:
    t1, t2, t3, t4 = st.tabs(["Aba 1", "Aba 2", "Aba 3", "Aba 4"])
    for i, aba in enumerate([t1, t2, t3, t4], start=1):
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


# ============================================================================
# Folium (mapa + camadas)
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
    try:
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google", name="Google Satellite", overlay=False, control=False, opacity=tiles_opacity
        ).add_to(m)
    except Exception:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery", name="Esri Satellite", overlay=False, control=False, opacity=tiles_opacity
        ).add_to(m)
    return m


def add_admin_outline(m, gdf, layer_name: str, color="#000000", weight=1.0):
    """
    Desenha:
      • 1) camada de LINHA (boundary) para o visual
      • 2) camada de ÁREA invisível para tooltip legível (hover na área inteira)
    """
    if gdf is None or folium is None:
        return

    # descobrir coluna de rótulo (nome/código) pelo layer
    cols_lower = {c.lower(): c for c in gdf.columns}
    label_col: Optional[str] = None
    lname = layer_name.lower()
    if "distr" in lname and "ds_nome" in cols_lower:
        label_col = cols_lower["ds_nome"]
    elif "subpref" in lname and "sp_nome" in cols_lower:
        label_col = cols_lower["sp_nome"]
    elif "setor" in lname:
        for k in ("cd_geocodi", "cd_setor"):
            if k in cols_lower:
                label_col = cols_lower[k]
                break

    # 1) LINHA (boundary)
    try:
        gf_line = gdf[["geometry"]].copy()
        gf_line["geometry"] = gf_line.geometry.boundary
        gf_line["geometry"] = gf_line.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)

        folium.GeoJson(
            data=gf_line.to_json(),
            name=f"{layer_name} (contorno)",
            pane="vectors",
            style_function=lambda f: {"fillOpacity": 0, "color": color, "weight": weight},
        ).add_to(m)
    except Exception as exc:
        st.warning(f"Falha ao preparar contorno: {exc}")
        return

    # 2) ÁREA (preenchimento transparente) — só para tooltip grande e hover confortável
    if label_col:
        try:
            gf_area = gdf[[label_col, "geometry"]].copy()
            gf_area["geometry"] = gf_area.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)

            folium.GeoJson(
                data=gf_area.to_json(),
                name=f"{layer_name} (hover)",
                pane="vectors",
                style_function=lambda f: {
                    "fillOpacity": 0.05,   # levemente visível para o hover "grudar"
                    "opacity": 0.0,        # sem stroke
                    "weight": 0,
                    "color": "#00000000",   # transparente
                },
                highlight_function=lambda f: {"weight": 2, "color": PB_COLORS["teal"], "fillOpacity": 0.10},
                tooltip=folium.features.GeoJsonTooltip(
                    fields=[label_col],
                    aliases=[""],          # sem prefixo
                    sticky=True,
                    labels=False,
                    style=(
                        "background-color: white; color: #111;"
                        "font-size: 16px; font-weight: 700;"
                        "border: 1px solid #222; border-radius: 8px;"
                        "padding: 6px 10px;"
                    ),
                    max_width=400,
                ),
            ).add_to(m)
        except Exception as exc:
            st.warning(f"Falha ao criar tooltip de área: {exc}")


def plot_density_variable(m, setores_gdf, dens_df):
    """Choropleth de `densidade_hec` por Setor Censitário (inteiro)."""
    if folium is None or gpd is None or setores_gdf is None or dens_df is None:
        return

    def _real(df, name):  # pega o nome real da coluna ignorando case
        return next((c for c in df.columns if c.lower() == name.lower()), name)

    # chave preferencial
    key = None
    for cand in ("CD_GEOCODI", "CD_SETOR"):
        if cand.lower() in [c.lower() for c in setores_gdf.columns] and cand.lower() in [c.lower() for c in dens_df.columns]:
            key = cand
            break
    if key is None:
        st.info("Não foi possível identificar a coluna de setor (CD_GEOCODI/CD_SETOR).")
        return

    key_g = _real(setores_gdf, key)
    key_d = _real(dens_df, key)
    val_d = _real(dens_df, "densidade_hec")
    if val_d not in dens_df.columns:
        st.info("Coluna `densidade_hec` não encontrada no Parquet de densidade.")
        return

    gdf = setores_gdf[[key_g, "geometry"]].rename(columns={key_g: "setor_key"})
    df = dens_df[[key_d, val_d]].rename(columns={key_d: "setor_key", val_d: "densidade_hec"})
    df["densidade_hec"] = pd.to_numeric(df["densidade_hec"], errors="coerce").round(0).astype("Int64")
    mg = gdf.merge(df, on="setor_key", how="inner").dropna(subset=["densidade_hec"])
    if mg.empty:
        st.info("Nenhum setor para desenhar após o merge.")
        return

    try:
        mg["geometry"] = mg.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    except Exception:
        pass

    q = mg["densidade_hec"].astype(float).quantile([0, .2, .4, .6, .8, 1]).tolist()
    palette = ["#F4DD63", "#B1BF7C", "#6FA097", "#D58243", "#14407D"]

    def style_fn(feat):
        v = feat["properties"].get("densidade_hec")
        idx = sum(float(v) >= x for x in q[:-1]) if v is not None else 0
        return {"fillOpacity": 0.65, "weight": 0.6, "color": "#ffffff",
                "fillColor": palette[min(idx, len(palette) - 1)]}

    try:
        folium.GeoJson(
            data=mg.to_json(),
            name="Densidade",
            pane="vectors",
            style_function=style_fn,
            tooltip=folium.features.GeoJsonTooltip(
                fields=["setor_key", "densidade_hec"],
                aliases=["Setor:", "Densidade (hab/ha):"],
                sticky=True,
                labels=True,
                style=(
                    "background-color: white; color: #111;"
                    "font-size: 16px; font-weight: 600;"
                    "border: 1px solid #222; border-radius: 8px;"
                    "padding: 6px 10px;"
                ),
                max_width=420,
            ),
        ).add_to(m)
    except Exception as exc:
        st.warning(f"Falha ao desenhar densidade: {exc}")


def bar_for_density(dens_df: Optional[pd.DataFrame]):
    if dens_df is None or dens_df.empty:
        st.info("Sem dados de densidade para o gráfico.")
        return
    col = next((c for c in dens_df.columns if c.lower() == "densidade_hec"), None)
    if col is None:
        st.info("Coluna `densidade_hec` não encontrada no Parquet.")
        return
    vals = pd.to_numeric(dens_df[col], errors="coerce").dropna().round(0).astype(int)
    st.subheader("Densidade (hab/ha) — distribuição")
    fig = px.histogram(vals, nbins=30, height=540)
    fig.update_traces(marker_color=PB_COLORS["laranja"])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# App
# ============================================================================
def build_tabs():  # (mantém placeholders das abas)
    t1, t2, t3, t4 = st.tabs(["Aba 1", "Aba 2", "Aba 3", "Aba 4"])
    for i, aba in enumerate([t1, t2, t3, t4], start=1):
        with aba:
            st.markdown(f"**Conteúdo da Aba {i}** — espaço para textos/explicações.")
    st.write("")


def main() -> None:
    inject_css()
    build_header(get_logo_path())
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
    dens_df = load_density() if variavel == "Densidade" else None

    # Debug útil para validar leitura/paths
    st.caption(f"ADM_DIR: {ADM_DIR}")
    st.caption(f"DENS_DIR: {DENS_DIR}")
    if ADM_DIR.exists():
        st.caption(f"Arquivos em ADM_DIR: {[p.name for p in ADM_DIR.glob('*.parquet')]}")
    if DENS_DIR.exists():
        st.caption(f"Arquivos em DENS_DIR: {[p.name for p in DENS_DIR.glob('*.parquet')]}")

    if gdf_limite is not None:
        st.caption(f"{limite}: {len(gdf_limite)} feições | colunas: {list(gdf_limite.columns)[:10]}")
    if dens_df is not None:
        st.caption(f"Densidade DF: {len(dens_df)} linhas | colunas: {list(dens_df.columns)[:10]}")

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


# Evita “tela branca”: mostra exceções em tela
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.exception(e)
