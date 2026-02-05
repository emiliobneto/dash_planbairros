# app.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import base64
from typing import Optional, Dict, Any, Tuple, Set

import streamlit as st

# Map stack
try:
    import geopandas as gpd  # type: ignore
    import folium  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
except Exception:
    gpd = None  # type: ignore
    folium = None  # type: ignore
    st_folium = None  # type: ignore


# =============================================================================
# Config / identidade (UI)
# =============================================================================
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
PB_NAVY = PB_COLORS["navy"]

SIMPLIFY_TOL = 0.0006  # simplificação só para LINHAS/outlines
CARTO_LIGHT_URL = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
CARTO_ATTR = "© OpenStreetMap contributors © CARTO"

REPO_ROOT = Path.cwd()
DATA_DIR = REPO_ROOT / "limites_administrativos"
LOGO_PATH = REPO_ROOT / "assets" / "logo_todos.jpg"
LOGO_HEIGHT = 46


# =============================================================================
# Paths (ajuste se seus arquivos tiverem outros nomes)
# =============================================================================
P_SUBPREF = DATA_DIR / "Subprefeitura.parquet"
P_DIST = DATA_DIR / "Distritos.parquet"
P_ISO = DATA_DIR / "Isocronas.parquet"
P_QUADRA = DATA_DIR / "Quadras.parquet"
P_LOTE = DATA_DIR / "Lotes.parquet"


# =============================================================================
# IDs EXPLÍCITOS (conforme seu encadeamento)
# =============================================================================
# IDs únicos por camada
SUBPREF_ID = "subpref_id"
DIST_ID = "distrito_id"
ISO_ID = "iso_id"
QUADRA_ID = "quadra_id"
LOTE_ID = "lote_id"

# Chaves pai → filho (FK)
DIST_PARENT = "subpref_id"      # Distritos têm subpref_id
ISO_PARENT = "distrito_id"      # Isócronas têm distrito_id
QUADRA_PARENT = "iso_id"        # Quadras têm iso_id (recomendado)
LOTE_PARENT = "quadra_id"       # Lotes têm quadra_id


# =============================================================================
# CSS (estética principal)
# =============================================================================
def _logo_data_uri() -> str:
    if LOGO_PATH.exists():
        suf = LOGO_PATH.suffix.lstrip(".").lower()
        mime = "jpeg" if suf in ("jpg", "jpeg") else suf
        b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
        return f"data:image/{mime};base64,{b64}"
    return (
        "https://raw.githubusercontent.com/streamlit/brand/refs/heads/main/"
        "logomark/streamlit-mark-color.png"
    )


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700;900&display=swap');
        html, body, .stApp {{
            font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
        }}
        .main .block-container {{
            padding-top: .15rem !important;
            padding-bottom: .6rem !important;
        }}
        .pb-row {{ display:flex; align-items:center; gap:12px; margin-bottom:0; }}
        .pb-logo {{ height:{LOGO_HEIGHT}px; width:auto; display:block; border-radius:8px; }}
        .pb-header {{
            background:{PB_NAVY}; color:#fff; border-radius:14px;
            padding:14px 15px; width:100%;
        }}
        .pb-title {{ font-size:2.25rem; font-weight:900; line-height:1.05; letter-spacing:.2px; }}
        .pb-subtitle {{ font-size:1.05rem; opacity:.95; margin-top:5px; }}

        .pb-card {{
            background:#fff;
            border:1px solid rgba(20,64,125,.10);
            box-shadow:0 1px 2px rgba(0,0,0,.04);
            border-radius:14px;
            padding:9px;
        }}
        .pb-badge {{
            display:inline-flex; align-items:center; gap:8px;
            padding:6px 10px; border-radius:999px;
            border:1px solid rgba(20,64,125,.12);
            background:#fff; color:#111; font-size:12px; font-weight:700;
        }}
        .pb-note {{ font-size:12px; color:rgba(17,17,17,.75); line-height:1.35; }}
        .pb-divider {{ height:1px; background:rgba(20,64,125,.10); margin:10px 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        f"""
        <div class="pb-header">
          <div class="pb-row">
            <img src="{_logo_data_uri()}" class="pb-logo" />
            <div style="display:flex;flex-direction:column">
              <div class="pb-title">PlanBairros</div>
              <div class="pb-subtitle">Plataforma de visualização e planejamento em escala de bairro</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Estado (drill-down)
# =============================================================================
LEVELS = ["subpref", "distrito", "isocrona", "quadra", "lote"]


def init_state() -> None:
    st.session_state.setdefault("level", "subpref")
    st.session_state.setdefault("selected_subpref_id", None)
    st.session_state.setdefault("selected_distrito_id", None)
    st.session_state.setdefault("selected_iso_ids", set())     # multi
    st.session_state.setdefault("selected_quadra_ids", set())  # multi
    st.session_state.setdefault("view_center", (-23.55, -46.63))
    st.session_state.setdefault("view_zoom", 11)


def reset_to(level: str) -> None:
    st.session_state["level"] = level
    if level == "subpref":
        st.session_state["selected_subpref_id"] = None
        st.session_state["selected_distrito_id"] = None
        st.session_state["selected_iso_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["view_center"] = (-23.55, -46.63)
        st.session_state["view_zoom"] = 11
    elif level == "distrito":
        st.session_state["selected_distrito_id"] = None
        st.session_state["selected_iso_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
    elif level == "isocrona":
        st.session_state["selected_iso_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
    elif level == "quadra":
        st.session_state["selected_quadra_ids"] = set()


def _back_one_level() -> None:
    cur = st.session_state["level"]
    idx = LEVELS.index(cur)
    if idx == 0:
        return
    reset_to(LEVELS[idx - 1])


def _toggle_in_set(key: str, value: Any) -> None:
    s: Set[Any] = st.session_state.get(key, set())
    if value in s:
        s.remove(value)
    else:
        s.add(value)
    st.session_state[key] = s


# =============================================================================
# Leitura/saneamento (sem análise, só preparação)
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def read_layer_parquet(path: Path) -> Optional["gpd.GeoDataFrame"]:
    if gpd is None:
        return None
    if not path.exists():
        return None
    gdf = gpd.read_parquet(path)
    try:
        if gdf.crs is None:
            gdf = gdf.set_crs(4326, allow_override=True)
        else:
            gdf = gdf.to_crs(4326)
    except Exception:
        pass
    return gdf


def _drop_bad_geoms(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty:
        return gdf
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notna()]
    try:
        gdf = gdf[~gdf.geometry.is_empty]
    except Exception:
        pass
    return gdf


def _simplify_lines(gdf: "gpd.GeoDataFrame", tol: float) -> "gpd.GeoDataFrame":
    """Simplificação apenas para linhas/outlines."""
    if gdf is None or gdf.empty:
        return gdf
    gdf = gdf.copy()
    try:
        gdf["geometry"] = gdf.geometry.simplify(tol, preserve_topology=True)
    except Exception:
        pass
    return _drop_bad_geoms(gdf)


def bounds_center_zoom(gdf: "gpd.GeoDataFrame") -> Tuple[Tuple[float, float], int]:
    minx, miny, maxx, maxy = gdf.total_bounds
    center = ((miny + maxy) / 2, (minx + maxx) / 2)
    dx = maxx - minx
    if dx < 0.03:
        z = 15
    elif dx < 0.08:
        z = 14
    elif dx < 0.15:
        z = 13
    elif dx < 0.30:
        z = 12
    else:
        z = 11
    return center, z


def pick_feature_id(gdf: "gpd.GeoDataFrame", click_latlon: Dict[str, float], id_col: str) -> Optional[Any]:
    """Identifica feição clicada via point-in-polygon (rápido se gdf já recortado)."""
    if gdf is None or gdf.empty or not click_latlon:
        return None
    if id_col not in gdf.columns:
        return None

    lat = click_latlon.get("lat")
    lng = click_latlon.get("lng")
    if lat is None or lng is None:
        return None

    try:
        from shapely.geometry import Point  # type: ignore
        pt = Point(lng, lat)

        cand = gdf
        try:
            if hasattr(gdf, "sindex") and gdf.sindex is not None:
                idx = list(gdf.sindex.intersection(pt.bounds))
                if idx:
                    cand = gdf.iloc[idx]
        except Exception:
            pass

        hit = cand[cand.geometry.contains(pt)]
        if hit.empty:
            hit = cand[cand.geometry.intersects(pt)]
        if hit.empty:
            return None
        return hit.iloc[0][id_col]
    except Exception:
        return None


def subset_by_parent(child: "gpd.GeoDataFrame", parent_col: str, parent_val: Any) -> "gpd.GeoDataFrame":
    """Filtro por FK (encadeamento rápido)."""
    if child is None or child.empty:
        return child
    child = _drop_bad_geoms(child)
    if parent_col not in child.columns or parent_val is None:
        return child.iloc[0:0].copy()
    try:
        return child[child[parent_col] == parent_val]
    except Exception:
        return child.iloc[0:0].copy()


def subset_by_parent_multi(child: "gpd.GeoDataFrame", parent_col: str, parent_vals: Set[Any]) -> "gpd.GeoDataFrame":
    """Filtro por FK multi (encadeamento rápido)."""
    if child is None or child.empty:
        return child
    child = _drop_bad_geoms(child)
    if parent_col not in child.columns or not parent_vals:
        return child.iloc[0:0].copy()
    try:
        return child[child[parent_col].isin(list(parent_vals))]
    except Exception:
        return child.iloc[0:0].copy()


# =============================================================================
# Folium – Carto + panes + camadas
# =============================================================================
def make_carto_map(center=(-23.55, -46.63), zoom=11):
    if folium is None:
        return None

    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True, prefer_canvas=True)

    folium.TileLayer(
        tiles=CARTO_LIGHT_URL,
        attr=CARTO_ATTR,
        name="Carto Positron",
        overlay=False,
        control=False,
        subdomains="abcd",
        max_zoom=20,
    ).add_to(m)

    # panes: shapes abaixo, linhas acima
    try:
        folium.map.CustomPane("detail_shapes", z_index=640).add_to(m)
        folium.map.CustomPane("top_lines", z_index=660).add_to(m)
    except Exception:
        pass

    return m


def add_outline(m, gdf: "gpd.GeoDataFrame", name: str, color="#111111", weight=1.2, show=True) -> None:
    """Outline (linhas) — padrão do seu código modelo (boundary + simplify)."""
    if folium is None or gdf is None or gdf.empty:
        return

    line = gdf[["geometry"]].copy()
    try:
        line["geometry"] = line.geometry.boundary
    except Exception:
        return
    line = _simplify_lines(line, SIMPLIFY_TOL)

    fg = folium.FeatureGroup(name=name, show=show)
    folium.GeoJson(
        data=line.to_json(),
        pane="top_lines",
        style_function=lambda _f: {"fillOpacity": 0, "color": color, "weight": weight},
    ).add_to(fg)
    fg.add_to(m)


def add_polygons_selectable(
    m,
    gdf: "gpd.GeoDataFrame",
    name: str,
    id_col: str,
    selected_ids: Optional[Set[Any]] = None,
    pane: str = "detail_shapes",
    base_color: str = "#111111",
    base_weight: float = 0.8,
    fill_color: str = "#ffffff",
    fill_opacity: float = 0.08,
    selected_color: str = "#14407D",
    selected_weight: float = 2.4,
    selected_fill_opacity: float = 0.18,
) -> None:
    """Polígonos com estilo diferenciado para selecionados (UX/UI)."""
    if folium is None or gdf is None or gdf.empty:
        return
    if id_col not in gdf.columns:
        # sem id -> render sem seleção
        id_col = ""

    selected_ids = selected_ids or set()

    cols = ["geometry"] + ([id_col] if id_col else [])
    mini = gdf[cols].copy()

    def style_fn(feat):
        props = feat.get("properties", {})
        fid = props.get(id_col) if id_col else None
        is_sel = (fid in selected_ids) if fid is not None else False
        return {
            "color": (selected_color if is_sel else base_color),
            "weight": (selected_weight if is_sel else base_weight),
            "fillColor": fill_color,
            "fillOpacity": (selected_fill_opacity if is_sel else fill_opacity),
        }

    fg = folium.FeatureGroup(name=name, show=True)
    folium.GeoJson(
        data=mini.to_json(),
        pane=pane,
        style_function=style_fn,
        highlight_function=lambda _f: {"weight": base_weight + 1.2, "fillOpacity": min(fill_opacity + 0.10, 0.35)},
    ).add_to(fg)
    fg.add_to(m)


# =============================================================================
# UI lateral (breadcrumb + status)
# =============================================================================
def left_panel() -> None:
    st.markdown("<span class='pb-badge'>🧭 Fluxo</span>", unsafe_allow_html=True)
    st.caption("Subprefeitura → Distrito → Isócronas → Quadras → Lotes")

    st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.button("Voltar 1 nível", use_container_width=True, on_click=_back_one_level)
    with c2:
        st.button("Reset", type="secondary", use_container_width=True, on_click=reset_to, args=("subpref",))

    st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)

    st.markdown("<span class='pb-badge'>📌 Seleção</span>", unsafe_allow_html=True)
    st.write(
        {
            "level": st.session_state["level"],
            "subpref_id": st.session_state["selected_subpref_id"],
            "distrito_id": st.session_state["selected_distrito_id"],
            "iso_ids": sorted(list(st.session_state["selected_iso_ids"]))[:30],
            "quadra_ids": sorted(list(st.session_state["selected_quadra_ids"]))[:30],
        }
    )
    st.markdown(
        "<div class='pb-note'>Clique no mapa para avançar. Em Isócronas e Quadras, o clique alterna seleção (multi).</div>",
        unsafe_allow_html=True,
    )


def kpis_row() -> None:
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        st.markdown("<span class='pb-badge'>📍 Nível</span>", unsafe_allow_html=True)
        st.markdown(f"**{st.session_state['level']}**")
        st.caption("Camada em exibição")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        st.markdown("<span class='pb-badge'>🏛️ Subpref</span>", unsafe_allow_html=True)
        st.markdown(f"**{st.session_state['selected_subpref_id'] or '—'}**")
        st.caption("Seleção")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        st.markdown("<span class='pb-badge'>🗺️ Distrito</span>", unsafe_allow_html=True)
        st.markdown(f"**{st.session_state['selected_distrito_id'] or '—'}**")
        st.caption("Seleção")
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        st.markdown("<span class='pb-badge'>🧩 Multi</span>", unsafe_allow_html=True)
        st.markdown(f"**Iso: {len(st.session_state['selected_iso_ids'])} | Quad: {len(st.session_state['selected_quadra_ids'])}**")
        st.caption("Seleções ativas")
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# App
# =============================================================================
def main() -> None:
    init_state()
    inject_css()
    render_header()

    if gpd is None or folium is None or st_folium is None:
        st.error("Este app requer `geopandas`, `folium` e `streamlit-folium`.")
        return

    left, right = st.columns([1, 4], gap="large")

    with left:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        left_panel()
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        kpis_row()
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)

        # --- Carrega camadas (cacheadas) ---
        g_subpref = read_layer_parquet(P_SUBPREF)
        g_dist = read_layer_parquet(P_DIST)
        g_iso = read_layer_parquet(P_ISO)
        g_quadra = read_layer_parquet(P_QUADRA)
        g_lote = read_layer_parquet(P_LOTE)

        # --- Validação mínima ---
        if g_subpref is None or g_subpref.empty:
            st.warning(f"Subprefeitura não encontrada/vazia: {P_SUBPREF}")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        # saneia
        g_subpref = _drop_bad_geoms(g_subpref)
        if g_dist is not None:
            g_dist = _drop_bad_geoms(g_dist)
        if g_iso is not None:
            g_iso = _drop_bad_geoms(g_iso)
        if g_quadra is not None:
            g_quadra = _drop_bad_geoms(g_quadra)
        if g_lote is not None:
            g_lote = _drop_bad_geoms(g_lote)

        # --- Mapa base ---
        m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])
        if m is None:
            st.error("Falha ao inicializar o mapa.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        level = st.session_state["level"]

        # =====================================================================================
        # NÍVEL 0: SUBPREF (abre com LINHAS)
        # =====================================================================================
        if level == "subpref":
            st.markdown("### Subprefeituras")
            st.caption("Clique em uma subprefeitura para abrir os distritos (recorte).")

            add_outline(m, g_subpref, "Subprefeituras (linha)", color="#111111", weight=1.25, show=True)

            st.session_state["view_center"] = (-23.55, -46.63)
            st.session_state["view_zoom"] = 11

        # =====================================================================================
        # NÍVEL 1: DISTRITOS (da subpref selecionada)
        # =====================================================================================
        elif level == "distrito":
            if g_dist is None or g_dist.empty:
                st.warning(f"Distritos não encontrados/vazios: {P_DIST}")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            sp = st.session_state["selected_subpref_id"]
            if sp is None:
                reset_to("subpref")
                st.rerun()

            g_show = subset_by_parent(g_dist, DIST_PARENT, sp)

            st.markdown(f"### Distritos (Subpref {sp})")
            st.caption("Clique em um distrito para abrir isócronas.")

            add_outline(m, g_show, "Distritos (linha)", color="#111111", weight=1.0, show=True)
            add_polygons_selectable(
                m,
                g_show,
                name="Distritos (clique)",
                id_col=DIST_ID,
                selected_ids=set(),
                base_color="#111111",
                base_weight=0.6,
                fill_color="#ffffff",
                fill_opacity=0.06,
            )

            if not g_show.empty:
                center, zoom = bounds_center_zoom(g_show)
                st.session_state["view_center"], st.session_state["view_zoom"] = center, zoom
                m.location, m.zoom_start = center, zoom

        # =====================================================================================
        # NÍVEL 2: ISÓCRONAS (do distrito selecionado)
        # =====================================================================================
        elif level == "isocrona":
            if g_iso is None or g_iso.empty:
                st.warning(f"Isócronas não encontradas/vazias: {P_ISO}")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            d = st.session_state["selected_distrito_id"]
            if d is None:
                reset_to("distrito")
                st.rerun()

            g_show = subset_by_parent(g_iso, ISO_PARENT, d)

            st.markdown(f"### Isócronas (Distrito {d})")
            st.caption("Clique para selecionar múltiplas isócronas. Ao selecionar, abre Quadras.")

            add_polygons_selectable(
                m,
                g_show,
                name="Isócronas (multi)",
                id_col=ISO_ID,
                selected_ids=st.session_state["selected_iso_ids"],
                base_color="#111111",
                base_weight=0.6,
                fill_color="#ffffff",
                fill_opacity=0.10,
                selected_color=PB_NAVY,
                selected_weight=2.4,
                selected_fill_opacity=0.20,
            )
            add_outline(m, g_show, "Isócronas (linha)", color="#111111", weight=0.7, show=False)

            if not g_show.empty:
                center, zoom = bounds_center_zoom(g_show)
                st.session_state["view_center"], st.session_state["view_zoom"] = center, zoom
                m.location, m.zoom_start = center, zoom

        # =====================================================================================
        # NÍVEL 3: QUADRAS (das isócronas selecionadas)
        # =====================================================================================
        elif level == "quadra":
            if g_quadra is None or g_quadra.empty:
                st.warning(f"Quadras não encontradas/vazias: {P_QUADRA}")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            iso_ids: Set[Any] = st.session_state["selected_iso_ids"]
            if not iso_ids:
                reset_to("isocrona")
                st.rerun()

            g_show = subset_by_parent_multi(g_quadra, QUADRA_PARENT, iso_ids)

            st.markdown("### Quadras (Isócronas selecionadas)")
            st.caption("Clique para selecionar múltiplas quadras. Ao selecionar, abre Lotes.")

            add_polygons_selectable(
                m,
                g_show,
                name="Quadras (multi)",
                id_col=QUADRA_ID,
                selected_ids=st.session_state["selected_quadra_ids"],
                base_color="#111111",
                base_weight=0.45,
                fill_color="#ffffff",
                fill_opacity=0.10,
                selected_color=PB_NAVY,
                selected_weight=2.2,
                selected_fill_opacity=0.22,
            )

            if not g_show.empty:
                center, zoom = bounds_center_zoom(g_show)
                st.session_state["view_center"], st.session_state["view_zoom"] = center, min(zoom + 1, 17)
                m.location, m.zoom_start = st.session_state["view_center"], st.session_state["view_zoom"]

        # =====================================================================================
        # NÍVEL 4: LOTES (das quadras selecionadas) — último nível
        # =====================================================================================
        else:  # "lote"
            if g_lote is None or g_lote.empty:
                st.warning(f"Lotes não encontrados/vazios: {P_LOTE}")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            quad_ids: Set[Any] = st.session_state["selected_quadra_ids"]
            if not quad_ids:
                reset_to("quadra")
                st.rerun()

            g_show = subset_by_parent_multi(g_lote, LOTE_PARENT, quad_ids)

            st.markdown("### Lotes (Quadras selecionadas)")
            st.caption("Último nível de agregação. Aqui você pode plugar painel de atributos e ações.")

            add_polygons_selectable(
                m,
                g_show,
                name="Lotes",
                id_col=LOTE_ID if (LOTE_ID in g_show.columns) else "",
                selected_ids=set(),
                base_color="#111111",
                base_weight=0.25,
                fill_color="#ffffff",
                fill_opacity=0.10,
            )

            if not g_show.empty:
                center, zoom = bounds_center_zoom(g_show)
                st.session_state["view_center"], st.session_state["view_zoom"] = center, min(zoom + 1, 18)
                m.location, m.zoom_start = st.session_state["view_center"], st.session_state["view_zoom"]

        # Layer control
        try:
            folium.LayerControl(position="bottomright", collapsed=False).add_to(m)
        except Exception:
            pass

        # Render
        out = st_folium(m, height=780, use_container_width=True, key="map_view", returned_objects=[])

        # =====================================================================================
        # CLIQUES: atualiza estado e avança no fluxo
        # =====================================================================================
        click = (out or {}).get("last_clicked")
        if click:
            # 0) subpref -> distrito
            if level == "subpref":
                picked = pick_feature_id(g_subpref, click, SUBPREF_ID)
                if picked is not None:
                    st.session_state["selected_subpref_id"] = picked
                    st.session_state["level"] = "distrito"
                    st.rerun()

            # 1) distrito -> isocrona
            elif level == "distrito":
                sp = st.session_state["selected_subpref_id"]
                g_show = subset_by_parent(g_dist, DIST_PARENT, sp)  # type: ignore
                picked = pick_feature_id(g_show, click, DIST_ID)
                if picked is not None:
                    st.session_state["selected_distrito_id"] = picked
                    st.session_state["selected_iso_ids"] = set()
                    st.session_state["selected_quadra_ids"] = set()
                    st.session_state["level"] = "isocrona"
                    st.rerun()

            # 2) isocrona (multi): toggle; se houver >=1 -> quadra
            elif level == "isocrona":
                d = st.session_state["selected_distrito_id"]
                g_show = subset_by_parent(g_iso, ISO_PARENT, d)  # type: ignore
                picked = pick_feature_id(g_show, click, ISO_ID)
                if picked is not None:
                    _toggle_in_set("selected_iso_ids", picked)
                    if len(st.session_state["selected_iso_ids"]) >= 1:
                        st.session_state["selected_quadra_ids"] = set()
                        st.session_state["level"] = "quadra"
                    st.rerun()

            # 3) quadra (multi): toggle; se houver >=1 -> lote
            elif level == "quadra":
                iso_ids = st.session_state["selected_iso_ids"]
                g_show = subset_by_parent_multi(g_quadra, QUADRA_PARENT, iso_ids)  # type: ignore
                picked = pick_feature_id(g_show, click, QUADRA_ID)
                if picked is not None:
                    _toggle_in_set("selected_quadra_ids", picked)
                    if len(st.session_state["selected_quadra_ids"]) >= 1:
                        st.session_state["level"] = "lote"
                    st.rerun()

            # 4) lote: (opcional) selecionar lote aqui — por enquanto só visual
            else:
                pass

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
