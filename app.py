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
            .pb-card h4 {{ font-size: 2.0rem !important; margin: 0 0 .6rem 0; }}
            .pb-card label, .pb-card .stMarkdown p {{ font-size: 1.9rem !important; font-weight: 800 !important; }}
            .pb-card div[role="combobox"] {{ font-size: 1.7rem !important; min-height: 58px !important; }}
            .pb-card [data-baseweb="select"] * {{ font-size: 1.7rem !important; }}
            .pb-card .stSelectbox svg {{ transform: scale(1.7); }}

            .stTabs [data-baseweb="tab-list"] button[role="tab"] {{
                background:transparent; border-bottom:3px solid transparent; font-weight:700; font-size: 1.1rem;
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
DENS_DIR = _resolve_dir("densidade")


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
    "SetoresCensitarios2023": ["SetoresCensitarios2023", "SetoresCensitarios"],
    "ZonasOD2023": ["ZonasOD2023", "ZonasOD"],
    "Subprefeitura": ["Subprefeitura", "Subprefeituras", "subprefeitura"],
}
DENSITY_CANDIDATES = ["Densidade", "Densidade2023", "Desindade", "Desindade2023", "HabitacaoPrecaria"]


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
    if gpd is None:
        st.info("Geopandas não disponível — instale `geopandas`, `shapely` e `pyarrow`.")
        return None
    path = _first_parquet_matching(ADM_DIR, ADMIN_NAME_MAP.get(layer_name, [layer_name]))
    if path is None:
        st.warning(f"Arquivo Parquet não encontrado para '{layer_name}' em {ADM_DIR}.")
        return None
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
    try:
        gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)
    except Exception:
        pass
    return gdf


@st.cache_data(show_spinner=False)
def load_density() -> Optional[pd.DataFrame]:
    search: list[Path] = []
    direct = DENS_DIR.with_suffix(".parquet")
    if direct.exists():
        search.append(direct)
    for base in (DENS_DIR, ADM_DIR):
        if base.exists():
            by_name = _first_parquet_matching(base, DENSITY_CANDIDATES)
            if by_name is not None:
                search.append(by_name)
            search += sorted(base.glob("*.parquet"))
    seen = set()
    for p in search:
        if p in seen:
            continue
        seen.add(p)
        df = _safe_read_parquet(p)
        if df is None:
            continue
        if any(c.lower() == "densidade_hec" for c in df.columns):
            return df
    st.info("Dados de densidade não encontrados com coluna `densidade_hec`.")
    return None


# ============================================================================
# UI helpers
# ============================================================================
def build_tabs() -> None:
    a1, a2, a3, a4 = st.tabs(["Aba 1", "Aba 2", "Aba 3", "Aba 4"])
    for i, aba in enumerate([a1, a2, a3, a4], start=1):
        with aba:
            st.markdown(f"**Conteúdo da Aba {i}** — espaço para textos/explicações.")
    st.write("")


def build_controls(key_prefix="main_") -> Tuple[str, str, str, str, bool]:
    st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
    st.markdown("<h4>Configurações</h4>", unsafe_allow_html=True)

    limite = st.selectbox(
        "Limites Administrativos",
        ["Distritos", "SetoresCensitarios2023", "ZonasOD2023", "Subprefeitura"],
        index=0,
        key=f"{key_prefix}pb_limite",
        help="Escolha a desagregação espacial para exibir como contorno.",
    )
    variavel = st.selectbox(
        "Variáveis",
        ["Densidade", "Zoneamento"],
        index=1,
        key=f"{key_prefix}pb_variavel",
        help="Variáveis agregadas por Setor Censitário (densidade/zoneamento).",
    )
    metrica = st.selectbox("Métricas", ["—"], index=0, key=f"{key_prefix}pb_metrica")
    info = st.selectbox("Informações", ["—"], index=0, key=f"{key_prefix}pb_info")
    labels_on = st.checkbox(
        "Rótulos",
        value=False,
        key=f"{key_prefix}pb_labels_on",
        help="Mostra nomes fixos nos centróides (tamanho dinâmico por zoom).",
    )

    st.markdown("</div>", unsafe_allow_html=True)
    return limite, variavel, metrica, info, labels_on


# ============================================================================
# Folium
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

    # Esri Imagery (sem rótulos)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri Satellite",
        overlay=False,
        control=False,
        opacity=tiles_opacity,
    ).add_to(m)
    return m


def inject_leaflet_css(m, font_px: int = 900):
    """CSS dentro do mapa para tooltips gigantes (classe própria)."""
    if folium is None:
        return
    css = f"""
    <style>
      .leaflet-tooltip-pane, .leaflet-pane.leaflet-tooltip-pane, .leaflet-pane.leaflet-tooltip-pane * {{
        transform: none !important; -webkit-transform: none !important;
      }}
      .leaflet-tooltip.pb-big-tooltip,
      .leaflet-tooltip.pb-big-tooltip * {{
        font-size: {font_px}px !important;
        font-weight: 900 !important;
        color: #111 !important;
        line-height: 1 !important;
      }}
      .leaflet-tooltip.pb-big-tooltip {{
        background: #fff !important;
        border: 3px solid #222 !important; border-radius: 12px !important;
        padding: 26px 34px !important; white-space: nowrap !important;
        pointer-events: none !important; box-shadow: 0 2px 6px rgba(0,0,0,.25) !important;
        z-index: 200000 !important;
      }}
    </style>
    """
    from folium import Element  # type: ignore

    m.get_root().html.add_child(Element(css))


def inject_label_scaler(m, min_px=16, max_px=26, min_zoom=9, max_zoom=18):
    """
    Redimensiona '.pb-static-label' conforme zoom do Leaflet.
    - min_px/max_px: tamanho da fonte (px) em min_zoom/max_zoom
    - min_zoom/max_zoom: faixa de zoom onde ocorre a transição
    """
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
          var t = (z - minZ) / (maxZ - minZ);   // 0..1
          var px = Math.round(minPx + t * (maxPx - minPx));
          return px;
        }}
        function updatePBLabels() {{
          var z = {map_var}.getZoom();
          var px = scaleFont(z);
          var els = document.querySelectorAll('.pb-static-label');
          els.forEach(function(el) {{
            el.style.fontSize = px + 'px';
          }});
        }}
        {map_var}.on('zoomend', updatePBLabels);
        {map_var}.whenReady(updatePBLabels);
      }})();
    </script>
    """
    m.get_root().html.add_child(Element(js))


def add_admin_outline(m, gdf, layer_name: str, color="#000000", weight=1.2):
    if gdf is None or folium is None:
        return
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

    # Linha (contorno)
    gf_line = gdf[["geometry"]].copy()
    gf_line["geometry"] = gf_line.geometry.boundary
    gf_line["geometry"] = gf_line.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    folium.GeoJson(
        data=gf_line.to_json(),
        name=f"{layer_name} (contorno)",
        pane="vectors",
        style_function=lambda f: {"fillOpacity": 0, "color": color, "weight": weight},
    ).add_to(m)

    # Área para hover (tooltip grande)
    if label_col:
        gf_area = gdf[[label_col, "geometry"]].copy()
        gf_area["geometry"] = gf_area.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
        folium.GeoJson(
            data=gf_area.to_json(),
            name=f"{layer_name} (hover)",
            pane="vectors",
            style_function=lambda f: {"fillOpacity": 0.05, "opacity": 0, "weight": 0, "color": "#00000000"},
            highlight_function=lambda f: {"weight": 2, "color": PB_COLORS["teal"], "fillOpacity": 0.10},
            tooltip=folium.features.GeoJsonTooltip(
                fields=[label_col],
                aliases=[""],
                sticky=True,
                labels=False,
                class_name="pb-big-tooltip",
                style=(
                    "background:#fff;color:#111;font-size:64px;font-weight:800;white-space:nowrap;"
                    "border:1px solid #222;border-radius:8px;padding:10px 14px;"
                ),
                max_width=1200,
            ),
        ).add_to(m)


def add_centroid_labels(m, gdf, label_col: str):
    """
    Rótulos permanentes com classe 'pb-static-label'.
    O tamanho será controlado dinamicamente por 'inject_label_scaler'.
    """
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
            "style=\"font: 500 12px/1 Roboto, -apple-system, Segoe UI, Helvetica, Arial, sans-serif;"
            "color:#111; text-shadow:0 0 2px #fff, 0 0 6px #fff; white-space:nowrap;\">"
            f"{name}</div>"
        )
        folium.Marker(
            location=[pt.y, pt.x],
            icon=folium.DivIcon(html=html, icon_size=(0, 0), icon_anchor=(0, 0)),
            z_index_offset=1000,
        ).add_to(m)


def plot_density_variable(m, setores_gdf, dens_df):
    if folium is None or gpd is None or setores_gdf is None or dens_df is None:
        return

    def _real(df, name):
        return next((c for c in df.columns if c.lower() == name.lower()), name)

    key = None
    for cand in ("CD_GEOCODI", "CD_SETOR"):
        if cand.lower() in [c.lower() for c in setores_gdf.columns] and cand.lower() in [
            c.lower() for c in dens_df.columns
        ]:
            key = cand
            break
    if key is None:
        st.info("Não foi possível identificar a coluna de setor (CD_GEOCODI/CD_SETOR).")
        return
    key_g = _real(setores_gdf, key)
    key_d = _real(dens_df, key)
    val_d = _real(dens_df, "densidade_hec")
    if val_d not in dens_df.columns:
        st.info("Coluna `densidade_hec` não encontrada.")
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

    q = mg["densidade_hec"].astype(float).quantile([0, 0.2, 0.4, 0.6, 0.8, 1]).tolist()
    palette = ["#F4DD63", "#B1BF7C", "#6FA097", "#D58243", "#14407D"]

    def style_fn(feat):
        v = feat["properties"].get("densidade_hec")
        idx = sum(float(v) >= x for x in q[:-1]) if v is not None else 0
        return {
            "fillOpacity": 0.65,
            "weight": 0.6,
            "color": "#ffffff",
            "fillColor": palette[min(idx, len(palette) - 1)],
        }

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
            class_name="pb-big-tooltip",
            style=(
                "background:#fff;color:#111;font-size:64px;font-weight:800;white-space:nowrap;"
                "border:1px solid #222;border-radius:8px;padding:10px 14px;"
            ),
            max_width=1200,
        ),
    ).add_to(m)


def bar_for_density(dens_df: Optional[pd.DataFrame]):
    if dens_df is None or dens_df.empty:
        st.info("Sem dados de densidade para o gráfico.")
        return
    col = next((c for c in dens_df.columns if c.lower() == "densidade_hec"), None)
    if col is None:
        st.info("Coluna `densidade_hec` não encontrada.")
        return
    vals = pd.to_numeric(dens_df[col], errors="coerce").dropna().round(0).astype(int)
    st.subheader("Densidade (hab/ha) — distribuição")
    fig = px.histogram(vals, nbins=30, height=520)
    fig.update_traces(marker_color=PB_COLORS["laranja"])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# App
# ============================================================================
def _safe_rerun() -> None:
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def main() -> None:
    inject_css()

    # Header
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

    build_tabs()

    pre_show_chart = st.session_state.get("_show_chart", False)

    with st.container():
        if pre_show_chart:
            left, map_col, right = st.columns([1, 5, 1], gap="small")
        else:
            left, map_col = st.columns([1, 6], gap="small")
            right = None

        with left:
            limite, variavel, _, _, labels_on = build_controls("main_")

        gdf_limite = load_admin_layer(limite)
        dens_df = load_density() if variavel == "Densidade" else None
        show_chart = (variavel == "Densidade") and (dens_df is not None)
        if show_chart != pre_show_chart:
            st.session_state["_show_chart"] = show_chart
            _safe_rerun()

        with map_col:
            if folium is not None and st_folium is not None:
                center = (-23.55, -46.63)
                if gdf_limite is not None and len(gdf_limite) > 0:
                    minx, miny, maxx, maxy = gdf_limite.total_bounds
                    center = ((miny + maxy) / 2, (minx + maxx) / 2)
                height = 850 if right is None else 700

                fmap = make_satellite_map(center=center, zoom=10, tiles_opacity=0.5)

                # Tooltip gigante (hover) e escala dinâmica dos rótulos estáticos
                inject_leaflet_css(fmap, font_px=900)
                # <<< Ajuste aqui a faixa de tamanhos/zooms (16px quando afastado):
                inject_label_scaler(fmap, min_px=16, max_px=26, min_zoom=9, max_zoom=18)

                if fmap is not None:
                    if gdf_limite is not None:
                        add_admin_outline(fmap, gdf_limite, layer_name=limite, color="#000000", weight=1.2)

                        # rótulos permanentes (dinâmicos por zoom)
                        if labels_on:
                            cols_lower = {c.lower(): c for c in gdf_limite.columns}
                            label_col = None
                            if "ds_nome" in cols_lower:
                                label_col = cols_lower["ds_nome"]
                            elif "sp_nome" in cols_lower:
                                label_col = cols_lower["sp_nome"]
                            elif "cd_geocodi" in cols_lower:
                                label_col = cols_lower["cd_geocodi"]
                            elif "cd_setor" in cols_lower:
                                label_col = cols_lower["cd_setor"]
                            if label_col:
                                add_centroid_labels(fmap, gdf_limite, label_col)

                    if show_chart:
                        setores = load_admin_layer("SetoresCensitarios2023")
                        if setores is not None:
                            plot_density_variable(fmap, setores, dens_df)

                    st_folium(fmap, use_container_width=True, height=height)
            else:
                st.info("Para o mapa satélite, instale `folium` e `streamlit-folium`.")

        if right is not None and show_chart:
            with right:
                bar_for_density(dens_df)


if __name__ == "__main__":
    main()
