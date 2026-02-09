# app.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import base64
import re
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
# CONFIG / IDENTIDADE (UI)
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

SIMPLIFY_TOL = 0.0006
CARTO_LIGHT_URL = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
CARTO_ATTR = "© OpenStreetMap contributors © CARTO"

REPO_ROOT = Path.cwd()
ASSETS_DIR = REPO_ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "logo_todos.jpg"
LOGO_HEIGHT = 46

# =============================================================================
# GOOGLE DRIVE (PASTA)
# =============================================================================
GOOGLE_DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1YdJuACMnudDgb6__wAl_TYibvnxlJz0M?usp=drive_link"

DATA_CACHE_DIR = REPO_ROOT / "data_cache"
DRIVE_SYNC_DIR = DATA_CACHE_DIR / "drive_folder_sync"

# nomes exatos conforme seu print
EXPECTED_FILES = {
    "subpref": "Subprefeitura.parquet",
    "dist": "Distrito.parquet",
    "iso": "Isocronas.parquet",
    "quadra": "Quadras.parquet",
    "lote": "Lotes.parquet",
    "censo": "Setorcensitario.parquet",
}


def _extract_drive_folder_id(url: str) -> Optional[str]:
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None


@st.cache_resource(show_spinner=False)
def ensure_drive_folder_synced(folder_url: str, out_dir: Path) -> Path:
    """
    Baixa a pasta do Google Drive para disco local.
    Cacheada em memória do processo (não rebaixa a cada rerun).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import gdown  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Dependência ausente: 'gdown'. Adicione em requirements.txt: gdown\n"
            f"Erro: {e}"
        )

    # Se já tem conteúdo, não rebaixa
    if any(out_dir.rglob("*")):
        return out_dir

    folder_id = _extract_drive_folder_id(folder_url)
    if not folder_id:
        raise RuntimeError("Não consegui extrair o folder_id do link do Google Drive.")

    url = f"https://drive.google.com/drive/folders/{folder_id}"

    # download_folder pode criar subpasta interna; depois a gente procura com rglob
    gdown.download_folder(
        url=url,
        output=str(out_dir),
        quiet=True,
        use_cookies=False,
        remaining_ok=True,
    )
    return out_dir


def resolve_expected_files(root: Path) -> Dict[str, Path]:
    """
    Procura cada arquivo esperado recursivamente dentro do root sincronizado.
    """
    found: Dict[str, Path] = {}
    for key, fname in EXPECTED_FILES.items():
        hits = list(root.rglob(fname))
        if hits:
            found[key] = hits[0]
    return found


# =============================================================================
# IDs EXPLÍCITOS (encadeamento)
# =============================================================================
SUBPREF_ID = "subpref_id"
DIST_ID = "distrito_id"
ISO_ID = "iso_id"
QUADRA_ID = "quadra_id"
LOTE_ID = "lote_id"
CENSO_ID = "censo_id"

DIST_PARENT = "subpref_id"
ISO_PARENT = "distrito_id"
QUADRA_PARENT = "iso_id"
LOTE_PARENT = "quadra_id"
CENSO_PARENT = "iso_id"


# =============================================================================
# CSS (estética)
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
# ESTADO (drill-down)
# =============================================================================
LEVELS = ["subpref", "distrito", "isocrona", "quadra", "final"]


def init_state() -> None:
    st.session_state.setdefault("level", "subpref")
    st.session_state.setdefault("selected_subpref_id", None)
    st.session_state.setdefault("selected_distrito_id", None)
    st.session_state.setdefault("selected_iso_ids", set())     # multi
    st.session_state.setdefault("selected_quadra_ids", set())  # multi
    st.session_state.setdefault("final_mode", "lote")          # "lote" | "censo"
    st.session_state.setdefault("view_center", (-23.55, -46.63))
    st.session_state.setdefault("view_zoom", 11)


def reset_to(level: str) -> None:
    st.session_state["level"] = level
    if level == "subpref":
        st.session_state["selected_subpref_id"] = None
        st.session_state["selected_distrito_id"] = None
        st.session_state["selected_iso_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["final_mode"] = "lote"
        st.session_state["view_center"] = (-23.55, -46.63)
        st.session_state["view_zoom"] = 11
    elif level == "distrito":
        st.session_state["selected_distrito_id"] = None
        st.session_state["selected_iso_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["final_mode"] = "lote"
    elif level == "isocrona":
        st.session_state["selected_iso_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["final_mode"] = "lote"
    elif level == "quadra":
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["final_mode"] = "lote"


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
# IO / SANITIZAÇÃO
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=64)
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
# MAPA (Folium)
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

    try:
        folium.map.CustomPane("detail_shapes", z_index=640).add_to(m)
        folium.map.CustomPane("top_lines", z_index=660).add_to(m)
    except Exception:
        pass

    return m


def add_outline(m, gdf: "gpd.GeoDataFrame", name: str, color="#111111", weight=1.2, show=True) -> None:
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
    if folium is None or gdf is None or gdf.empty:
        return
    if id_col not in gdf.columns:
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
        highlight_function=lambda _f: {
            "weight": base_weight + 1.2,
            "fillOpacity": min(fill_opacity + 0.10, 0.35),
        },
    ).add_to(fg)
    fg.add_to(m)


# =============================================================================
# UI
# =============================================================================
def left_panel(files_found: Dict[str, Path]) -> None:
    st.markdown("<span class='pb-badge'>🧭 Fluxo</span>", unsafe_allow_html=True)
    st.caption("Subprefeitura → Distrito → Isócronas → Quadras → (Lotes | Setor)")

    st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.button("Voltar 1 nível", use_container_width=True, on_click=_back_one_level)
    with c2:
        st.button("Reset", type="secondary", use_container_width=True, on_click=reset_to, args=("subpref",))

    if st.session_state["level"] == "final":
        st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)
        st.markdown("<span class='pb-badge'>🧩 Nível final</span>", unsafe_allow_html=True)
        st.session_state["final_mode"] = st.radio(
            "Visualizar",
            ["lote", "censo"],
            index=0 if st.session_state["final_mode"] == "lote" else 1,
            horizontal=True,
        )

    st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)
    st.markdown("<span class='pb-badge'>📌 Seleção</span>", unsafe_allow_html=True)
    st.write(
        {
            "level": st.session_state["level"],
            "subpref_id": st.session_state["selected_subpref_id"],
            "distrito_id": st.session_state["selected_distrito_id"],
            "iso_ids": sorted(list(st.session_state["selected_iso_ids"]))[:25],
            "quadra_ids": sorted(list(st.session_state["selected_quadra_ids"]))[:25],
        }
    )
    st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)
    st.markdown("<span class='pb-badge'>🗂️ Arquivos detectados</span>", unsafe_allow_html=True)
    st.write({k: str(v) for k, v in files_found.items()})


def kpis_row() -> None:
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        st.markdown("<span class='pb-badge'>📍 Nível</span>", unsafe_allow_html=True)
        st.markdown(f"**{st.session_state['level']}**")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        st.markdown("<span class='pb-badge'>🏛️ Subpref</span>", unsafe_allow_html=True)
        st.markdown(f"**{st.session_state['selected_subpref_id'] or '—'}**")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        st.markdown("<span class='pb-badge'>🗺️ Distrito</span>", unsafe_allow_html=True)
        st.markdown(f"**{st.session_state['selected_distrito_id'] or '—'}**")
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        st.markdown("<span class='pb-badge'>🧩 Multi</span>", unsafe_allow_html=True)
        st.markdown(f"**Iso: {len(st.session_state['selected_iso_ids'])} | Quad: {len(st.session_state['selected_quadra_ids'])}**")
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# APP
# =============================================================================
def main() -> None:
    init_state()
    inject_css()
    render_header()

    if gpd is None or folium is None or st_folium is None:
        st.error("Este app requer `geopandas`, `folium` e `streamlit-folium`.")
        return

    # 1) Sync pasta do Drive
    with st.spinner("Baixando dados do Google Drive…"):
        try:
            drive_dir = ensure_drive_folder_synced(GOOGLE_DRIVE_FOLDER_URL, DRIVE_SYNC_DIR)
        except Exception as e:
            st.error("Falha ao baixar a pasta do Google Drive.")
            st.code(str(e))
            st.info("Confira permissões: 'Qualquer pessoa com o link' (Leitor).")
            return

    files_found = resolve_expected_files(drive_dir)

    # 2) Validação rápida
    missing = [k for k in EXPECTED_FILES.keys() if k not in files_found]
    if missing:
        st.error("Não encontrei alguns arquivos esperados dentro da pasta baixada.")
        st.write("Faltando:", missing)
        st.write("Procurando por:", EXPECTED_FILES)
        st.write("Pasta sincronizada:", str(drive_dir))
        st.write("Arquivos encontrados:", {k: str(v) for k, v in files_found.items()})
        return

    # 3) Carrega (cache_data)
    g_subpref = read_layer_parquet(files_found["subpref"])
    g_dist = read_layer_parquet(files_found["dist"])
    g_iso = read_layer_parquet(files_found["iso"])
    g_quadra = read_layer_parquet(files_found["quadra"])
    g_lote = read_layer_parquet(files_found["lote"])
    g_censo = read_layer_parquet(files_found["censo"])

    if g_subpref is None or g_subpref.empty:
        st.error(f"Subprefeitura vazia/erro ao ler: {files_found['subpref']}")
        return

    # saneia
    g_subpref = _drop_bad_geoms(g_subpref)
    g_dist = _drop_bad_geoms(g_dist) if g_dist is not None else None
    g_iso = _drop_bad_geoms(g_iso) if g_iso is not None else None
    g_quadra = _drop_bad_geoms(g_quadra) if g_quadra is not None else None
    g_lote = _drop_bad_geoms(g_lote) if g_lote is not None else None
    g_censo = _drop_bad_geoms(g_censo) if g_censo is not None else None

    left, right = st.columns([1, 4], gap="large")

    with left:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        left_panel(files_found)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        kpis_row()
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)

        m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])
        if m is None:
            st.error("Falha ao inicializar o mapa.")
            return

        level = st.session_state["level"]

        if level == "subpref":
            st.markdown("### Subprefeituras")
            add_outline(m, g_subpref, "Subprefeituras (linha)", color="#111111", weight=1.25, show=True)
            st.session_state["view_center"] = (-23.55, -46.63)
            st.session_state["view_zoom"] = 11

        elif level == "distrito":
            sp = st.session_state["selected_subpref_id"]
            if sp is None:
                reset_to("subpref"); st.rerun()

            g_show = subset_by_parent(g_dist, DIST_PARENT, sp)  # type: ignore
            st.markdown(f"### Distritos (Subpref {sp})")
            add_outline(m, g_show, "Distritos (linha)", color="#111111", weight=1.0, show=True)
            add_polygons_selectable(m, g_show, "Distritos (clique)", DIST_ID, selected_ids=set(), base_weight=0.6, fill_opacity=0.06)

            if not g_show.empty:
                center, zoom = bounds_center_zoom(g_show)
                st.session_state["view_center"], st.session_state["view_zoom"] = center, zoom
                m.location, m.zoom_start = center, zoom

        elif level == "isocrona":
            d = st.session_state["selected_distrito_id"]
            if d is None:
                reset_to("distrito"); st.rerun()

            g_show = subset_by_parent(g_iso, ISO_PARENT, d)  # type: ignore
            st.markdown(f"### Isócronas (Distrito {d})")
            add_polygons_selectable(
                m, g_show, "Isócronas (multi)", ISO_ID,
                selected_ids=st.session_state["selected_iso_ids"],
                base_weight=0.6, fill_opacity=0.10,
                selected_color=PB_NAVY, selected_weight=2.4, selected_fill_opacity=0.20
            )

            if not g_show.empty:
                center, zoom = bounds_center_zoom(g_show)
                st.session_state["view_center"], st.session_state["view_zoom"] = center, zoom
                m.location, m.zoom_start = center, zoom

        elif level == "quadra":
            iso_ids: Set[Any] = st.session_state["selected_iso_ids"]
            if not iso_ids:
                reset_to("isocrona"); st.rerun()

            g_show = subset_by_parent_multi(g_quadra, QUADRA_PARENT, iso_ids)  # type: ignore
            st.markdown("### Quadras (Isócronas selecionadas)")
            add_polygons_selectable(
                m, g_show, "Quadras (multi)", QUADRA_ID,
                selected_ids=st.session_state["selected_quadra_ids"],
                base_weight=0.45, fill_opacity=0.10,
                selected_color=PB_NAVY, selected_weight=2.2, selected_fill_opacity=0.22
            )

            if not g_show.empty:
                center, zoom = bounds_center_zoom(g_show)
                st.session_state["view_center"], st.session_state["view_zoom"] = center, min(zoom + 1, 17)
                m.location, m.zoom_start = st.session_state["view_center"], st.session_state["view_zoom"]

        else:  # final
            iso_ids: Set[Any] = st.session_state["selected_iso_ids"]
            quad_ids: Set[Any] = st.session_state["selected_quadra_ids"]
            if not iso_ids:
                reset_to("isocrona"); st.rerun()

            mode = st.session_state["final_mode"]
            if mode == "lote":
                if not quad_ids:
                    reset_to("quadra"); st.rerun()
                g_show = subset_by_parent_multi(g_lote, LOTE_PARENT, quad_ids)  # type: ignore
                st.markdown("### Lotes (Quadras selecionadas)")
                add_polygons_selectable(m, g_show, "Lotes", LOTE_ID, selected_ids=set(), base_weight=0.25, fill_opacity=0.10)
            else:
                g_show = subset_by_parent_multi(g_censo, CENSO_PARENT, iso_ids)  # type: ignore
                st.markdown("### Setores Censitários (Isócronas selecionadas)")
                add_polygons_selectable(m, g_show, "Setor censitário", CENSO_ID, selected_ids=set(), base_weight=0.35, fill_opacity=0.08)

            if g_show is not None and not g_show.empty:
                center, zoom = bounds_center_zoom(g_show)
                st.session_state["view_center"], st.session_state["view_zoom"] = center, min(zoom + 1, 18)
                m.location, m.zoom_start = st.session_state["view_center"], st.session_state["view_zoom"]

        try:
            folium.LayerControl(position="bottomright", collapsed=False).add_to(m)
        except Exception:
            pass

        out = st_folium(m, height=780, use_container_width=True, key="map_view", returned_objects=[])

        # CLIQUES → avança
        click = (out or {}).get("last_clicked")
        if click:
            if level == "subpref":
                picked = pick_feature_id(g_subpref, click, SUBPREF_ID)
                if picked is not None:
                    st.session_state["selected_subpref_id"] = picked
                    st.session_state["level"] = "distrito"
                    st.rerun()

            elif level == "distrito":
                sp = st.session_state["selected_subpref_id"]
                g_show = subset_by_parent(g_dist, DIST_PARENT, sp)  # type: ignore
                picked = pick_feature_id(g_show, click, DIST_ID)
                if picked is not None:
                    st.session_state["selected_distrito_id"] = picked
                    st.session_state["selected_iso_ids"] = set()
                    st.session_state["selected_quadra_ids"] = set()
                    st.session_state["final_mode"] = "lote"
                    st.session_state["level"] = "isocrona"
                    st.rerun()

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

            elif level == "quadra":
                iso_ids = st.session_state["selected_iso_ids"]
                g_show = subset_by_parent_multi(g_quadra, QUADRA_PARENT, iso_ids)  # type: ignore
                picked = pick_feature_id(g_show, click, QUADRA_ID)
                if picked is not None:
                    _toggle_in_set("selected_quadra_ids", picked)
                    if len(st.session_state["selected_quadra_ids"]) >= 1:
                        st.session_state["level"] = "final"
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
