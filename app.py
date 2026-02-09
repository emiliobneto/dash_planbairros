# app.py
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Set
import base64
import re
import os

import streamlit as st

# geo
try:
    import geopandas as gpd  # type: ignore
    import folium  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
except Exception:
    gpd = None  # type: ignore
    folium = None  # type: ignore
    st_folium = None  # type: ignore


# =============================================================================
# CONFIG / UI
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
DATA_CACHE_DIR = REPO_ROOT / "data_cache"
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ASSETS_DIR = REPO_ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "logo_todos.jpg"
LOGO_HEIGHT = 46


# =============================================================================
# IDS DO SEU ENCADEAMENTO (VOCÊ VAI GERAR)
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

LEVELS = ["subpref", "distrito", "isocrona", "quadra", "final"]


# =============================================================================
# GOOGLE DRIVE LINKS/IDS (fallback no próprio arquivo)
# - Pode ser URL completa OU só o file_id
# =============================================================================
DEFAULT_DRIVE = {
    "subpref": "https://drive.google.com/file/d/1vPY34cQLCoGfADpyOJjL9pNCYkVrmSZA/view?usp=drive_link",
    "dist": "https://drive.google.com/file/d/1K-t2BiSHN_D8De0oCFxzGdrEMhnGnh10/view?usp=drive_link",
    "iso": "https://drive.google.com/file/d/18ukyzMiYQ6vMqrU6-ctaPFbXMPX9XS9i/view?usp=drive_link",
    "quadra": "https://drive.google.com/file/d/1XKAYLNdt82ZPNAE-rseSuuaCFmjfn8IP/view?usp=drive_link",
    "lote": "https://drive.google.com/file/d/1oTFAZff1mVAWD6KQTJSz45I6B6pi6ceP/view?usp=drive_link",
    "censo": "https://drive.google.com/file/d/1APp7fxT2mgTpegVisVyQwjTRWOPz6Rgn/view?usp=drive_link",
}


def extract_drive_file_id(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""

    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", s)
    if m:
        return m.group(1)

    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", s)
    if m:
        return m.group(1)

    if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", s):
        return s

    return s


def _get_cfg(secret_key: str, env_key: str, default_val: str) -> str:
    v = ""
    try:
        v = str(st.secrets.get(secret_key, "")).strip()
    except Exception:
        v = ""
    if not v:
        v = os.getenv(env_key, "").strip()
    if not v:
        v = (default_val or "").strip()
    return extract_drive_file_id(v)


GDRIVE_FILE_IDS = {
    "subpref": _get_cfg("PB_SUBPREF_FILE_ID", "PB_SUBPREF_FILE_ID", DEFAULT_DRIVE["subpref"]),
    "dist": _get_cfg("PB_DISTRITO_FILE_ID", "PB_DISTRITO_FILE_ID", DEFAULT_DRIVE["dist"]),
    "iso": _get_cfg("PB_ISOCRONAS_FILE_ID", "PB_ISOCRONAS_FILE_ID", DEFAULT_DRIVE["iso"]),
    "quadra": _get_cfg("PB_QUADRAS_FILE_ID", "PB_QUADRAS_FILE_ID", DEFAULT_DRIVE["quadra"]),
    "lote": _get_cfg("PB_LOTES_FILE_ID", "PB_LOTES_FILE_ID", DEFAULT_DRIVE["lote"]),
    "censo": _get_cfg("PB_CENSO_FILE_ID", "PB_CENSO_FILE_ID", DEFAULT_DRIVE["censo"]),
}

LOCAL_FILENAMES = {
    "subpref": "Subprefeitura.parquet",
    "dist": "Distrito.parquet",
    "iso": "Isocronas.parquet",
    "quadra": "Quadras.parquet",
    "lote": "Lotes.parquet",
    "censo": "Setorcensitario.parquet",
}


# =============================================================================
# CSS / HEADER
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
        .pb-divider {{ height:1px; background:rgba(20,64,125,.10); margin:10px 0; }}
        .pb-note {{ font-size:12px; color:rgba(17,17,17,.75); line-height:1.35; }}
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
# STATE
# =============================================================================
def init_state() -> None:
    st.session_state.setdefault("level", "subpref")
    st.session_state.setdefault("selected_subpref_id", None)
    st.session_state.setdefault("selected_distrito_id", None)
    st.session_state.setdefault("selected_iso_ids", set())     # multi
    st.session_state.setdefault("selected_quadra_ids", set())  # multi
    st.session_state.setdefault("final_mode", "lote")          # "lote" | "censo"
    st.session_state.setdefault("view_center", (-23.55, -46.63))
    st.session_state.setdefault("view_zoom", 11)
    # DEBUG toggle
    st.session_state.setdefault("debug", False)


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
# DOWNLOAD DRIVE (streaming)
# =============================================================================
def _ensure_file_ids_configured(keys: list[str]) -> Optional[str]:
    missing = [k for k in keys if not GDRIVE_FILE_IDS.get(k)]
    if missing:
        return (
            "Faltam IDs/links para: " + ", ".join(missing) + ".\n"
            "Configure em .streamlit/secrets.toml (PB_*_FILE_ID) ou no DEFAULT_DRIVE."
        )
    return None


@st.cache_resource(show_spinner=False)
def download_drive_file(file_id: str, dst: Path) -> Path:
    import requests

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst

    session = requests.Session()
    URL = "https://drive.google.com/uc?export=download"

    def get_confirm_token(resp) -> Optional[str]:
        for k, v in resp.cookies.items():
            if k.startswith("download_warning"):
                return v
        return None

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

    if response.status_code != 200:
        raise RuntimeError(f"Download falhou (status={response.status_code}).")

    total = int(response.headers.get("Content-Length", 0) or 0)
    chunk = 1024 * 1024  # 1MB

    prog = st.progress(0, text=f"Baixando {dst.name}…")
    downloaded = 0

    with open(dst, "wb") as f:
        for part in response.iter_content(chunk_size=chunk):
            if not part:
                continue
            f.write(part)
            downloaded += len(part)
            if total > 0:
                pct = min(int(downloaded * 100 / total), 100)
                prog.progress(pct, text=f"Baixando {dst.name}… {pct}%")
    prog.empty()
    return dst


def ensure_local_layer(layer_key: str) -> Path:
    file_id = (GDRIVE_FILE_IDS.get(layer_key, "") or "").strip()
    if not file_id:
        raise RuntimeError(f"ID do Drive não configurado para layer '{layer_key}'.")
    dst = DATA_CACHE_DIR / LOCAL_FILENAMES[layer_key]
    return download_drive_file(file_id, dst)


# =============================================================================
# READ
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def read_gdf_parquet(path: Path) -> Optional["gpd.GeoDataFrame"]:
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


def subset_by_parent(child: "gpd.GeoDataFrame", parent_col: str, parent_val: Any) -> "gpd.GeoDataFrame":
    if child is None or child.empty:
        return child
    child = _drop_bad_geoms(child)
    if parent_col not in child.columns or parent_val is None:
        return child.iloc[0:0].copy()
    return child[child[parent_col] == parent_val]


def subset_by_parent_multi(child: "gpd.GeoDataFrame", parent_col: str, parent_vals: Set[Any]) -> "gpd.GeoDataFrame":
    if child is None or child.empty:
        return child
    child = _drop_bad_geoms(child)
    if parent_col not in child.columns or not parent_vals:
        return child.iloc[0:0].copy()
    return child[child[parent_col].isin(list(parent_vals))]


# =============================================================================
# CLICK HITTEST
# =============================================================================
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
        highlight_function=lambda _f: {"weight": base_weight + 1.2, "fillOpacity": min(fill_opacity + 0.10, 0.35)},
    ).add_to(fg)
    fg.add_to(m)


# =============================================================================
# UI
# =============================================================================
def left_panel() -> None:
    st.markdown("<span class='pb-badge'>🧭 Fluxo</span>", unsafe_allow_html=True)
    st.caption("Subprefeitura → Distrito → Isócronas → Quadras → (Lotes | Setor)")

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
            "iso_ids_n": len(st.session_state["selected_iso_ids"]),
            "quadra_ids_n": len(st.session_state["selected_quadra_ids"]),
        }
    )

    st.session_state["debug"] = st.checkbox("Debug (mostrar clique e colunas)", value=st.session_state["debug"])

    if st.session_state["level"] == "final":
        st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)
        st.markdown("<span class='pb-badge'>🧩 Nível final</span>", unsafe_allow_html=True)
        st.session_state["final_mode"] = st.radio(
            "Visualizar",
            ["lote", "censo"],
            index=0 if st.session_state["final_mode"] == "lote" else 1,
            horizontal=True,
        )


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
        st.markdown(
            f"**Iso: {len(st.session_state['selected_iso_ids'])} | Quad: {len(st.session_state['selected_quadra_ids'])}**"
        )
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

    msg = _ensure_file_ids_configured(["subpref", "dist", "iso", "quadra"])
    if msg:
        st.error(msg)
        st.stop()

    left, right = st.columns([1, 4], gap="large")
    with left:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        left_panel()
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        kpis_row()
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)

        level = st.session_state["level"]
        m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])
        if m is None:
            st.error("Falha ao inicializar o mapa.")
            return

        # --------------------------
        # LOAD + RENDER POR NÍVEL
        # --------------------------
        if level == "subpref":
            st.markdown("### Subprefeituras")
            sub_path = ensure_local_layer("subpref")
            g_subpref = read_gdf_parquet(sub_path)
            if g_subpref is None or g_subpref.empty:
                st.error("Subprefeitura vazia/erro ao ler.")
                st.stop()
            g_subpref = _drop_bad_geoms(g_subpref)

            # DEBUG: colunas e tipo de geometria
            if st.session_state["debug"]:
                st.write("SUBPREF cols:", list(g_subpref.columns))
                st.write("SUBPREF geom_type:", g_subpref.geom_type.value_counts())
                if SUBPREF_ID in g_subpref.columns:
                    st.write("SUBPREF null subpref_id:", int(g_subpref[SUBPREF_ID].isna().sum()))

            add_outline(m, g_subpref, "Subprefeituras (linha)", color="#111111", weight=1.25, show=True)

        elif level == "distrito":
            sp = st.session_state["selected_subpref_id"]
            if sp is None:
                reset_to("subpref")
                st.rerun()

            dist_path = ensure_local_layer("dist")
            g_dist = read_gdf_parquet(dist_path)
            if g_dist is None or g_dist.empty:
                st.error("Distrito vazio/erro ao ler.")
                st.stop()
            g_dist = _drop_bad_geoms(g_dist)

            g_show = subset_by_parent(g_dist, DIST_PARENT, sp)
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
                reset_to("distrito")
                st.rerun()

            iso_path = ensure_local_layer("iso")
            g_iso = read_gdf_parquet(iso_path)
            if g_iso is None or g_iso.empty:
                st.error("Isócronas vazias/erro ao ler.")
                st.stop()
            g_iso = _drop_bad_geoms(g_iso)

            g_show = subset_by_parent(g_iso, ISO_PARENT, d)
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
                reset_to("isocrona")
                st.rerun()

            quadra_path = ensure_local_layer("quadra")
            g_quadra = read_gdf_parquet(quadra_path)
            if g_quadra is None or g_quadra.empty:
                st.error("Quadras vazias/erro ao ler.")
                st.stop()
            g_quadra = _drop_bad_geoms(g_quadra)

            g_show = subset_by_parent_multi(g_quadra, QUADRA_PARENT, iso_ids)
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

        else:
            iso_ids: Set[Any] = st.session_state["selected_iso_ids"]
            quad_ids: Set[Any] = st.session_state["selected_quadra_ids"]
            if not iso_ids:
                reset_to("isocrona")
                st.rerun()

            mode = st.session_state["final_mode"]
            if mode == "lote" and not quad_ids:
                reset_to("quadra")
                st.rerun()

            st.markdown("### Nível final")
            st.warning("Arquivos do nível final podem ser pesados. O download só ocorre ao clicar no botão abaixo.")
            go = st.button("⬇️ Carregar dados do nível final", type="primary")

            if go:
                if mode == "lote":
                    msg2 = _ensure_file_ids_configured(["lote"])
                    if msg2:
                        st.error(msg2)
                        st.stop()

                    lote_path = ensure_local_layer("lote")
                    g_lote = read_gdf_parquet(lote_path)
                    if g_lote is None or g_lote.empty:
                        st.error("Lotes vazios/erro ao ler.")
                        st.stop()
                    g_lote = _drop_bad_geoms(g_lote)

                    g_show = subset_by_parent_multi(g_lote, LOTE_PARENT, quad_ids)
                    st.markdown("#### Lotes (Quadras selecionadas)")
                    add_polygons_selectable(m, g_show, "Lotes", LOTE_ID, selected_ids=set(), base_weight=0.25, fill_opacity=0.10)

                else:
                    msg2 = _ensure_file_ids_configured(["censo"])
                    if msg2:
                        st.error(msg2)
                        st.stop()

                    censo_path = ensure_local_layer("censo")
                    g_censo = read_gdf_parquet(censo_path)
                    if g_censo is None or g_censo.empty:
                        st.error("Setores vazios/erro ao ler.")
                        st.stop()
                    g_censo = _drop_bad_geoms(g_censo)

                    g_show = subset_by_parent_multi(g_censo, CENSO_PARENT, iso_ids)
                    st.markdown("#### Setores Censitários (Isócronas selecionadas)")
                    add_polygons_selectable(m, g_show, "Setor censitário", CENSO_ID, selected_ids=set(), base_weight=0.35, fill_opacity=0.08)

                if g_show is not None and not g_show.empty:
                    center, zoom = bounds_center_zoom(g_show)
                    st.session_state["view_center"], st.session_state["view_zoom"] = center, min(zoom + 1, 18)
                    m.location, m.zoom_start = st.session_state["view_center"], st.session_state["view_zoom"]

        try:
            folium.LayerControl(position="bottomright", collapsed=False).add_to(m)
        except Exception:
            pass

        # ✅ AJUSTE PRINCIPAL: habilitar retorno de clique
        out = st_folium(
            m,
            height=780,
            use_container_width=True,
            key="map_view",
            returned_objects=["last_clicked"],  # <-- importante
        )

        click = (out or {}).get("last_clicked")

        # DEBUG: ver o out e o clique
        if st.session_state["debug"]:
            st.write("OUT:", out)
            st.write("CLICK:", click)

        if click:
            if level == "subpref":
                sub_path = ensure_local_layer("subpref")
                g_subpref = read_gdf_parquet(sub_path)
                if g_subpref is None or g_subpref.empty:
                    st.stop()
                g_subpref = _drop_bad_geoms(g_subpref)

                picked = pick_feature_id(g_subpref, click, SUBPREF_ID)
                if st.session_state["debug"]:
                    st.write("PICKED subpref_id:", picked)

                if picked is not None:
                    st.session_state["selected_subpref_id"] = picked
                    st.session_state["level"] = "distrito"
                    st.rerun()

            elif level == "distrito":
                sp = st.session_state["selected_subpref_id"]
                dist_path = ensure_local_layer("dist")
                g_dist = read_gdf_parquet(dist_path)
                if g_dist is None or g_dist.empty:
                    st.stop()
                g_dist = _drop_bad_geoms(g_dist)

                g_show = subset_by_parent(g_dist, DIST_PARENT, sp)
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
                iso_path = ensure_local_layer("iso")
                g_iso = read_gdf_parquet(iso_path)
                if g_iso is None or g_iso.empty:
                    st.stop()
                g_iso = _drop_bad_geoms(g_iso)

                g_show = subset_by_parent(g_iso, ISO_PARENT, d)
                picked = pick_feature_id(g_show, click, ISO_ID)
                if picked is not None:
                    _toggle_in_set("selected_iso_ids", picked)
                    st.session_state["selected_quadra_ids"] = set()
                    st.session_state["level"] = "quadra"
                    st.rerun()

            elif level == "quadra":
                iso_ids = st.session_state["selected_iso_ids"]
                quadra_path = ensure_local_layer("quadra")
                g_quadra = read_gdf_parquet(quadra_path)
                if g_quadra is None or g_quadra.empty:
                    st.stop()
                g_quadra = _drop_bad_geoms(g_quadra)

                g_show = subset_by_parent_multi(g_quadra, QUADRA_PARENT, iso_ids)
                picked = pick_feature_id(g_show, click, QUADRA_ID)
                if picked is not None:
                    _toggle_in_set("selected_quadra_ids", picked)
                    if len(st.session_state["selected_quadra_ids"]) >= 1:
                        st.session_state["level"] = "final"
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
