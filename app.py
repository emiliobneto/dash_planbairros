# -*- coding: utf-8 -*-
"""
PlanBairros — Front‑end Streamlit (limites administrativos + densidade/zoneamento)

Regras implementadas:
• Configurações (limpas):
  - Limites administrativos (primeiro): Distritos, SetoresCensitarios2023, ZonasOD2023, Subprefeitura
  - Variáveis: Densidade, Zoneamento
• Dados carregados de: dash_planbairros/limites_administrativos/*.parquet (GeoParquet)
• Fundo: Google Satellite (50% opacidade)
• Limites: exibidos apenas com LINHA (sem preenchimento)
• Densidade (por Setor Censitário):
  - Choropleth no mapa
  - Seleção por CLIQUE de setores para compor o “bairro”
  - Painel à direita mostra a MÉDIA da densidade dos setores selecionados
  - Gráfico de barras compara Seleção vs Geral
• Zoneamento (por Setor Censitário):
  - Mapa categórico (cores por categoria); sem métrica à direita por enquanto

Observação: mantenho EPSG:4326 (WGS84) para visualização web.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

# Mapa interativo
from streamlit_folium import st_folium
import folium
from branca.colormap import linear

# Geoespacial
import geopandas as gpd
from shapely.geometry import Point

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
# Fonte de dados
# ============================================================
BASE_DIR = Path("dash_planbairros/limites_administrativos")
ADMIN_OPTIONS = [
    "Distritos",
    "SetoresCensitarios2023",
    "ZonasOD2023",
    "Subprefeitura",
]


def load_vector_parquet(name: str) -> Optional[gpd.GeoDataFrame]:
    """Lê um GeoParquet de BASE_DIR e garante CRS=EPSG:4326."""
    fp = BASE_DIR / f"{name}.parquet"
    if not fp.exists():
        st.warning(f"Arquivo não encontrado: {fp}")
        return None
    try:
        gdf = gpd.read_parquet(fp)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        else:
            gdf = gdf.to_crs(epsg=4326)
        return gdf
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Falha ao ler {fp.name}: {exc}")
        return None


def load_densidade_layer() -> Optional[gpd.GeoDataFrame]:
    """Lê 'densidade.parquet' (por Setor Censitário) em BASE_DIR e garante CRS=4326.
    Requer uma coluna numérica de densidade e (opcional) uma categórica de zoneamento.
    """
    fp = BASE_DIR / "densidade.parquet"
    if not fp.exists():
        st.warning(f"Camada de densidade não encontrada: {fp}")
        return None
    try:
        gdf = gpd.read_parquet(fp)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        else:
            gdf = gdf.to_crs(epsg=4326)
        # ID único por setor para seleção
        if "_setor_id" not in gdf.columns:
            gdf["_setor_id"] = gdf.index.astype(str)
        return gdf
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Falha ao ler densidade.parquet: {exc}")
        return None

# ============================================================
# Helpers de estilo e assets
# ============================================================

def inject_css() -> None:
    """Injeta CSS e aplica Roboto como fonte global."""
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
            .pb-header {{
                background: var(--pb-navy);
                color: #fff;
                border-radius: 18px;
                padding: 20px 24px;
                min-height: 110px;
                display: flex;
                align-items: center;
                gap: 16px;
            }}
            .pb-title {{ font-size: 2.4rem; line-height: 1.15; font-weight: 700; letter-spacing: .5px; }}
            .pb-subtitle {{ opacity: .95; margin-top: 4px; font-size: 1.1rem; font-weight: 400; }}
            .stTabs [data-baseweb="tab-list"] button[role="tab"] {{ background: transparent; border-bottom: 3px solid transparent; color: #2b2b2b; font-weight: 600; }}
            .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{ border-bottom: 3px solid var(--pb-teal) !important; color: var(--pb-navy) !important; }}
            .pb-card {{ background: #ffffff; border: 1px solid rgba(20,64,125,.10); box-shadow: 0 1px 2px rgba(0,0,0,.04); border-radius: 16px; padding: 16px; }}
            .pb-card h4 {{ margin: 0 0 8px 0; font-size: 1.05rem; color: var(--pb-navy); }}
            footer {{visibility: hidden;}}
            #MainMenu {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_logo_path() -> Optional[str]:
    candidates = [
        "assets/logo_todos.jpg",
        "assets/logo_paleta.jpg",
        "logo_todos.jpg",
        "logo_paleta.jpg",
        "/mnt/data/logo_todos.jpg",
        "/mnt/data/logo_paleta.jpg",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None

# ============================================================
# Componentes de UI
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
                <div class="pb-header">
                    <div style="display:flex;flex-direction:column">
                        <div class="pb-title">PlanBairros</div>
                        <div class="pb-subtitle">Plataforma de visualização e planejamento em nível de bairro</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def build_tabs() -> None:
    aba1, aba2, aba3, aba4 = st.tabs(["Aba 1", "Aba 2", "Aba 3", "Aba 4"])
    for i, aba in enumerate([aba1, aba2, aba3, aba4], start=1):
        with aba:
            st.markdown(f"**Conteúdo da Aba {i}** — espaço reservado.")
    st.write("")


def build_controls() -> Tuple[str, str]:
    """Configurações limpas: primeiro Limites, depois Variáveis."""
    st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
    st.markdown("<h4>Configurações</h4>", unsafe_allow_html=True)

    admin_name = st.selectbox(
        "Limites administrativos",
        ADMIN_OPTIONS,
        index=0,
        help="Escolha a desagregação para sobreposição (linhas sem preenchimento).",
        key="pb_detalhe",
    )

    variavel = st.selectbox(
        "Variáveis",
        ["Densidade", "Zoneamento"],
        index=0,
        help="Variável agregada por Setor Censitário.",
        key="pb_variavel",
    )

    st.markdown("</div>", unsafe_allow_html=True)
    return admin_name, variavel

# ============================================================
# Mapa e lógica de seleção
# ============================================================

def render_map_with_selection(admin_name: str, variavel: str) -> Tuple[Optional[float], Optional[float], int]:
    """Renderiza mapa com:
    - Google Satellite (50% opacidade)
    - Limites administrativos (linhas)
    - DENSIDADE: choropleth + seleção por clique de Setores; retorna médias
    - ZONEAMENTO: coloração categórica (sem métrica por enquanto)
    """
    gdf_admin = load_vector_parquet(admin_name)
    gdf_den = load_densidade_layer()

    # Centro do mapa
    center_lat, center_lon = -23.55, -46.63
    try:
        base_gdf = gdf_den if gdf_den is not None else gdf_admin
        if base_gdf is not None and not base_gdf.empty:
            c = base_gdf.geometry.unary_union.centroid
            center_lat, center_lon = float(c.y), float(c.x)
    except Exception:
        pass

    # Mapa base sem tiles padrão
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=None)

    # Google Satellite (50%) — atenção a termos de uso
    folium.TileLayer(
        tiles="https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite",
        subdomains=["mt0", "mt1", "mt2", "mt3"],
        opacity=0.5,
        control=False,
    ).add_to(m)

    # Limites administrativos: LINHAS
    if gdf_admin is not None and not gdf_admin.empty:
        folium.GeoJson(
            data=gdf_admin.__geo_interface__,
            name=f"Limites — {admin_name}",
            style_function=lambda f: {"color": PB_COLORS["navy"], "weight": 2, "fill": False, "fillOpacity": 0},
        ).add_to(m)

    mean_sel, mean_total, n_sel = None, None, 0

    # Colunas candidatas
    den_col = None
    zone_col = None
    if gdf_den is not None and not gdf_den.empty:
        # densidade: tenta nomes comuns; fallback primeira numérica
        cand_cols = ["densidade", "DENSIDADE", "dens", "dens_media", "dens_med"]
        den_col = next((c for c in cand_cols if c in gdf_den.columns), None)
        if den_col is None:
            num_cols = [c for c in gdf_den.columns if c != "geometry" and pd.api.types.is_numeric_dtype(gdf_den[c])]
            den_col = num_cols[0] if num_cols else None
        # zoneamento: tenta nomes comuns
        cat_cols = ["zoneamento", "ZONEAMENTO", "zona", "classe", "uso"]
        zone_col = next((c for c in cat_cols if c in gdf_den.columns), None)

    if variavel == "Densidade" and gdf_den is not None and not gdf_den.empty and den_col is not None:
        vmin, vmax = float(gdf_den[den_col].min()), float(gdf_den[den_col].max())
        cmap = linear.YlOrRd_09.scale(vmin, vmax)
        cmap.caption = "Densidade"

        def style_fn(feat):
            v = feat["properties"].get(den_col, 0)
            return {"color": "#666", "weight": 0.8, "fill": True, "fillOpacity": 0.6, "fillColor": cmap(v)}

        tooltip_fields = [c for c in gdf_den.columns if c != "geometry"][:6]
        tooltip = folium.features.GeoJsonTooltip(fields=tooltip_fields) if tooltip_fields else None

        folium.GeoJson(
            data=gdf_den.__geo_interface__,
            name="Densidade (Setores)",
            style_function=style_fn,
            tooltip=tooltip,
        ).add_to(m)
        cmap.add_to(m)

        # Estado da seleção
        sel_key = "_setores_sel"
        if sel_key not in st.session_state:
            st.session_state[sel_key] = set()

        # 1º render: captura clique
        st_map = st_folium(m, height=560, use_container_width=True, returned_objects=["last_object_clicked"])    
        if st_map and st_map.get("last_object_clicked"):
            lat = st_map["last_object_clicked"]["lat"]
            lon = st_map["last_object_clicked"]["lng"]
            pt = Point(lon, lat)
            try:
                hit = gdf_den[gdf_den.geometry.contains(pt)]
                if not hit.empty:
                    sid = str(hit.iloc[0].get("_setor_id", hit.index[0]))
                    if sid in st.session_state[sel_key]:
                        st.session_state[sel_key].remove(sid)
                    else:
                        st.session_state[sel_key].add(sid)
            except Exception:
                pass

        # Destaca seleção
        if st.session_state[sel_key]:
            sel = gdf_den[gdf_den["_setor_id"].astype(str).isin(st.session_state[sel_key])]
            n_sel = len(sel)
            folium.GeoJson(
                data=sel.__geo_interface__,
                name="Selecionados",
                style_function=lambda f: {"color": PB_COLORS["teal"], "weight": 3, "fill": False, "fillOpacity": 0},
            ).add_to(m)
            mean_sel = float(sel[den_col].mean()) if n_sel else None
        mean_total = float(gdf_den[den_col].mean())

        # 2º render: com seleção destacada
        st_folium(m, height=560, use_container_width=True)

    elif variavel == "Zoneamento" and gdf_den is not None and not gdf_den.empty and zone_col is not None:
        # Paleta categórica
        cats = list(pd.Series(gdf_den[zone_col].astype(str)).unique())
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        color_map = {c: palette[i % len(palette)] for i, c in enumerate(cats)}

        def style_fn(feat):
            k = str(feat["properties"].get(zone_col))
            return {"color": "#666", "weight": 0.6, "fill": True, "fillOpacity": 0.55, "fillColor": color_map.get(k, "#cccccc")}

        tooltip_fields = [c for c in gdf_den.columns if c != "geometry"][:6]
        tooltip = folium.features.GeoJsonTooltip(fields=tooltip_fields) if tooltip_fields else None

        folium.GeoJson(
            data=gdf_den.__geo_interface__,
            name="Zoneamento (Setores)",
            style_function=style_fn,
            tooltip=tooltip,
        ).add_to(m)
        st_folium(m, height=560, use_container_width=True)

    else:
        # Sem dados suficientes: mostra só fundo + limites
        st_folium(m, height=560, use_container_width=True)

    return mean_sel, mean_total, n_sel

# ============================================================
# Layout principal
# ============================================================

def main() -> None:
    inject_css()
    logo_path = get_logo_path()

    # Header
    with st.container():
        build_header(logo_path)
    st.write("")

    # Abas de topo (mantidas como placeholder)
    build_tabs()

    # Grade principal
    try:
        left, center, right = st.columns([1, 2, 2], gap="large")
    except TypeError:
        left, center, right = st.columns([1, 2, 2])

    with left:
        admin_name, variavel = build_controls()

    with center:
        mean_sel, mean_total, n_sel = render_map_with_selection(admin_name, variavel)

    with right:
        st.subheader("Resumo")
        if variavel == "Densidade" and mean_total is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Densidade média (seleção)", f"{mean_sel:.2f}" if mean_sel is not None else "—")
            with c2:
                st.metric("Densidade média (geral)", f"{mean_total:.2f}")

            if mean_sel is not None:
                _df = pd.DataFrame({"grupo": ["Selecionado", "Geral"], "densidade": [mean_sel, mean_total]})
                fig = px.bar(_df, x="grupo", y="densidade", text="densidade", height=460)
                fig.update_traces(marker_color=[PB_COLORS["teal"], PB_COLORS["laranja"]])
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Clique nos Setores Censitários no mapa para construir o seu bairro e ver a média.")
        elif variavel == "Zoneamento":
            st.write("Selecione *Densidade* para ver métricas. Zoneamento é apenas visual por enquanto.")
        else:
            st.write("Carregando dados...")


if __name__ == "__main__":
    main()
