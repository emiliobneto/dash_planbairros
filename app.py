# app.py
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Set
import os
import re
import base64

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
    "telha": "#C65534",   # marrom/telha
    "teal": "#6FA097",
    "navy": "#14407D",
}
PB_NAVY = PB_COLORS["navy"]
PB_TELHA = PB_COLORS["telha"]

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
# IDS / COLUNAS (SEUS ARQUIVOS DEVEM TER EXATAMENTE ISSO)
# =============================================================================
SUBPREF_ID = "subpref_id"
DIST_ID = "distrito_id"
ISO_ID = "iso_id"
QUADRA_ID = "quadra_id"
LOTE_ID = "lote_id"
CENSO_ID = "censo_id"

# colunas PAI (repetidas no arquivo FILHO)
DIST_PARENT = "subpref_id"     # distritos.subpref_id
ISO_PARENT = "distrito_id"     # isocronas.distrito_id
QUADRA_PARENT = "iso_id"       # quadras.iso_id  ✅ (como você disse)
LOTE_PARENT = "quadra_id"      # lotes.quadra_id ✅
CENSO_PARENT = "iso_id"        # setores.iso_id  ✅

LEVELS = ["subpref", "distrito", "isocrona", "quadra", "detail"]

# =============================================================================
# GOOGLE DRIVE LINKS/IDS
# - você pode deixar no secrets.toml (recomendado)
# - ou usar env vars
# - ou deixar hardcoded aqui (fallback)
# =============================================================================
def _get_conf(key: str, fallback: str = "") -> str:
    v = ""
    try:
        v = str(st.secrets.get(key, "")).strip()
    except Exception:
        v = ""
    if not v:
        v = str(os.getenv(key, "")).strip()
    if not v:
        v = fallback.strip()
    return v


# 👉 se quiser, pode colar seus links aqui como fallback.
PB_SUBPREF_FILE = _get_conf("PB_SUBPREF_FILE_ID", "")
PB_DISTRITO_FILE = _get_conf("PB_DISTRITO_FILE_ID", "")
PB_ISOCRONAS_FILE = _get_conf("PB_ISOCRONAS_FILE_ID", "")
PB_QUADRAS_FILE = _get_conf("PB_QUADRAS_FILE_ID", "")
PB_LOTES_FILE = _get_conf("PB_LOTES_FILE_ID", "")
PB_CENSO_FILE = _get_conf("PB_CENSO_FILE_ID", "")

GDRIVE_FILES = {
    "subpref": PB_SUBPREF_FILE,
    "dist": PB_DISTRITO_FILE,
    "iso": PB_ISOCRONAS_FILE,
    "quadra": PB_QUADRAS_FILE,
    "lote": PB_LOTES_FILE,
    "censo": PB_CENSO_FILE,
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
            padding:10px;
        }}
        .pb-badge {{
            display:inline-flex; align-items:center; gap:8px;
            padding:6px 10px; border-radius:999px;
            border:1px solid rgba(20,64,125,.12);
            background:#fff; color:#111; font-size:12px; font-weight:800;
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
    if idx <= 0:
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
# GOOGLE DRIVE: extrair file_id de URL ou aceitar id puro
# =============================================================================
def extract_drive_id(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    # já é id puro?
    if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", s):
        return s
    m = re.search(r"/d/([a-zA-Z0-9_-]{20,})/", s)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]{20,})", s)
    if m:
        return m.group(1)
    # fallback: tenta achar “um id” no texto
    m = re.search(r"([a-zA-Z0-9_-]{20,})", s)
    return m.group(1) if m else ""


def _ensure_config(keys: list[str]) -> Optional[str]:
    missing = [k for k in keys if not extract_drive_id(GDRIVE_FILES.get(k, ""))]
    if missing:
        return "Faltam FILE_IDs (ou links válidos) para: " + ", ".join(missing)
    return None


@st.cache_resource(show_spinner=False)
def download_drive_file(file_link_or_id: str, dst: Path) -> Path:
    import requests

    file_id = extract_drive_id(file_link_or_id)
    if not file_id:
        raise RuntimeError("Link/ID do Google Drive inválido.")

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

    r = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(r)
    if token:
        r = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

    if r.status_code != 200:
        raise RuntimeError(f"Download falhou (status={r.status_code}).")

    total = int(r.headers.get("Content-Length", 0) or 0)
    chunk = 1024 * 1024  # 1MB

    prog = st.progress(0, text=f"Baixando {dst.name}…")
    downloaded = 0

    with open(dst, "wb") as f:
        for part in r.iter_content(chunk_size=chunk):
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
    link = GDRIVE_FILES.get(layer_key, "").strip()
    if not extract_drive_id(link):
        raise RuntimeError(f"FILE_ID/link inválido para layer '{layer_key}'.")
    dst = DATA_CACHE_DIR / LOCAL_FILENAMES[layer_key]
    return download_drive_file(link, dst)


# =============================================================================
# READ / UTILS
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def read_gdf_parquet(path: Path) -> Optional["gpd.GeoDataFrame"]:
    if gpd is None or not path.exists():
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
# CLICK HITTEST (polígonos)
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
# MAPA
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

    # panes: sombra abaixo / shapes / linhas acima
    try:
        folium.map.CustomPane("shadow", z_index=610).add_to(m)
        folium.map.CustomPane("shapes", z_index=640).add_to(m)
        folium.map.CustomPane("lines", z_index=670).add_to(m)
    except Exception:
        pass

    return m


def add_shadow_parent(m, gdf: "gpd.GeoDataFrame", name: str) -> None:
    """Sombra do nível acima: telha, opacidade bem baixa, borda tracejada fina."""
    if folium is None or gdf is None or gdf.empty:
        return

    mini = gdf[["geometry"]].copy()
    fg = folium.FeatureGroup(name=f"⟂ {name} (sombra)", show=True)

    folium.GeoJson(
        data=mini.to_json(),
        pane="shadow",
        style_function=lambda _f: {
            "color": PB_TELHA,
            "weight": 0.6,
            "opacity": 0.55,
            "dashArray": "2,6",
            "fillColor": PB_TELHA,
            "fillOpacity": 0.10,  # bem sutil (sombra)
        },
    ).add_to(fg)

    fg.add_to(m)


def add_outline_lines(m, gdf: "gpd.GeoDataFrame", name: str, color="#111111", weight=1.05, show=True) -> None:
    """Outline (boundary) com acabamento melhor (simplify)"""
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
        pane="lines",
        style_function=lambda _f: {"fillOpacity": 0, "color": color, "weight": weight, "opacity": 0.95},
    ).add_to(fg)
    fg.add_to(m)


def add_polygons_selectable(
    m,
    gdf: "gpd.GeoDataFrame",
    name: str,
    id_col: str,
    selected_ids: Optional[Set[Any]] = None,
    pane: str = "shapes",
    base_color: str = "#111111",
    base_weight: float = 0.7,
    fill_color: str = "#ffffff",
    fill_opacity: float = 0.06,
    selected_color: str = "#14407D",
    selected_weight: float = 2.2,
    selected_fill_opacity: float = 0.16,
    tooltip_label: Optional[str] = None,
) -> None:
    """Polígonos clicáveis (inclusive “transparentes” para capturar clique)."""
    if folium is None or gdf is None or gdf.empty:
        return
    if id_col not in gdf.columns:
        return

    selected_ids = selected_ids or set()
    mini = gdf[["geometry", id_col]].copy()

    def style_fn(feat):
        fid = feat.get("properties", {}).get(id_col)
        is_sel = fid in selected_ids
        return {
            "color": (selected_color if is_sel else base_color),
            "weight": (selected_weight if is_sel else base_weight),
            "opacity": 0.95,
            "fillColor": fill_color,
            "fillOpacity": (selected_fill_opacity if is_sel else fill_opacity),
        }

    fg = folium.FeatureGroup(name=name, show=True)
    gj = folium.GeoJson(
        data=mini.to_json(),
        pane=pane,
        style_function=style_fn,
        highlight_function=lambda _f: {"weight": base_weight + 1.1, "fillOpacity": min(fill_opacity + 0.12, 0.35)},
    )
    if tooltip_label:
        gj.add_child(folium.features.GeoJsonTooltip(fields=[id_col], aliases=[tooltip_label], sticky=True))
    gj.add_to(fg)
    fg.add_to(m)


# =============================================================================
# UI
# =============================================================================
def left_panel(level: str) -> None:
    st.markdown("<span class='pb-badge'>🧭 Fluxo</span>", unsafe_allow_html=True)
    st.caption("Subprefeitura → Distrito → Isócronas → Quadras → (Lotes + Setor Censitário)")

    st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.button("Voltar 1 nível", use_container_width=True, on_click=_back_one_level)
    with c2:
        st.button("Reset", type="secondary", use_container_width=True, on_click=reset_to, args=("subpref",))

    st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)
    st.markdown("<span class='pb-badge'>📌 Seleção</span>", unsafe_allow_html=True)
    st.json(
        {
            "level": level,
            "subpref_id": st.session_state["selected_subpref_id"],
            "distrito_id": st.session_state["selected_distrito_id"],
            "iso_ids_n": len(st.session_state["selected_iso_ids"]),
            "quadra_ids_n": len(st.session_state["selected_quadra_ids"]),
        }
    )

    if level == "isocrona":
        st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)
        st.button("➡️ Ir para Quadras", use_container_width=True, disabled=(len(st.session_state["selected_iso_ids"]) == 0),
                  on_click=lambda: st.session_state.update({"level": "quadra"}))
        st.button("Limpar seleção de Isócronas", type="secondary", use_container_width=True,
                  on_click=lambda: st.session_state.update({"selected_iso_ids": set()}))

    if level == "quadra":
        st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)
        st.button("➡️ Ir para Detalhes", use_container_width=True, disabled=(len(st.session_state["selected_quadra_ids"]) == 0),
                  on_click=lambda: st.session_state.update({"level": "detail"}))
        st.button("Limpar seleção de Quadras", type="secondary", use_container_width=True,
                  on_click=lambda: st.session_state.update({"selected_quadra_ids": set()}))

    st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)
    st.markdown("<span class='pb-badge'>⚙️ Performance</span>", unsafe_allow_html=True)
    st.markdown(
        "<div class='pb-note'>"
        "Os arquivos são baixados sob demanda. "
        "Lotes/Setor só baixam no nível <b>detail</b> ao clicar em carregar."
        "</div>",
        unsafe_allow_html=True,
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

    # mínimos para o fluxo (lote/censo só depois)
    msg = _ensure_config(["subpref", "dist", "iso", "quadra"])
    if msg:
        st.error(msg)
        st.stop()

    left, right = st.columns([1, 4], gap="large")
    with left:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        left_panel(st.session_state["level"])
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        kpis_row()
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)

        level = st.session_state["level"]

        # mapa base
        m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])
        if m is None:
            st.error("Falha ao inicializar o mapa.")
            return

        # contexto de clique
        click_gdf = None
        click_id_col = None

        # -------------------
        # SUBPREF
        # -------------------
        if level == "subpref":
            st.markdown("### Subprefeituras")

            g_sub = read_gdf_parquet(ensure_local_layer("subpref"))
            if g_sub is None or g_sub.empty:
                st.error("Subprefeitura vazia/erro ao ler.")
                st.stop()
            g_sub = _drop_bad_geoms(g_sub)

            # Visual: linhas
            add_outline_lines(m, g_sub, "Subprefeituras (linha)", color="#111111", weight=1.15, show=True)
            # Clique: polígonos quase invisíveis (captura clique)
            add_polygons_selectable(
                m, g_sub, "Subprefeituras (clique)", SUBPREF_ID,
                selected_ids=set(),
                base_weight=0.25,
                fill_opacity=0.01,  # invisível no visual, mas clicável
                tooltip_label="Subpref ID: ",
            )

            click_gdf = g_sub
            click_id_col = SUBPREF_ID

        # -------------------
        # DISTRITO
        # -------------------
        elif level == "distrito":
            sp = st.session_state["selected_subpref_id"]
            if sp is None:
                reset_to("subpref")
                st.rerun()

            g_dist = read_gdf_parquet(ensure_local_layer("dist"))
            g_sub = read_gdf_parquet(ensure_local_layer("subpref"))
            if g_dist is None or g_dist.empty:
                st.error("Distrito vazio/erro ao ler.")
                st.stop()
            g_dist = _drop_bad_geoms(g_dist)

            g_sub_sel = subset_by_parent(g_sub, SUBPREF_ID, sp) if g_sub is not None else None
            if g_sub_sel is not None and not g_sub_sel.empty:
                add_shadow_parent(m, g_sub_sel, "Subprefeitura selecionada")

            g_show = subset_by_parent(g_dist, DIST_PARENT, sp)
            st.markdown(f"### Distritos (Subpref {sp})")

            add_outline_lines(m, g_show, "Distritos (linha)", color="#111111", weight=1.0, show=True)
            add_polygons_selectable(
                m, g_show, "Distritos (clique)", DIST_ID,
                selected_ids=set(),
                base_weight=0.55, fill_opacity=0.06,
                tooltip_label="Distrito ID: ",
            )

            if not g_show.empty:
                center, zoom = bounds_center_zoom(g_show)
                st.session_state["view_center"], st.session_state["view_zoom"] = center, zoom
                m.location, m.zoom_start = center, zoom

            click_gdf = g_show
            click_id_col = DIST_ID

        # -------------------
        # ISOCRONA (multi)
        # -------------------
        elif level == "isocrona":
            d = st.session_state["selected_distrito_id"]
            if d is None:
                reset_to("distrito")
                st.rerun()

            g_iso = read_gdf_parquet(ensure_local_layer("iso"))
            g_dist = read_gdf_parquet(ensure_local_layer("dist"))
            if g_iso is None or g_iso.empty:
                st.error("Isócronas vazias/erro ao ler.")
                st.stop()
            g_iso = _drop_bad_geoms(g_iso)

            g_dist_sel = subset_by_parent(g_dist, DIST_ID, d) if g_dist is not None else None
            if g_dist_sel is not None and not g_dist_sel.empty:
                add_shadow_parent(m, g_dist_sel, "Distrito selecionado")

            g_show = subset_by_parent(g_iso, ISO_PARENT, d)
            st.markdown(f"### Isócronas (Distrito {d})")

            add_polygons_selectable(
                m, g_show, "Isócronas (multi)", ISO_ID,
                selected_ids=st.session_state["selected_iso_ids"],
                base_weight=0.55, fill_opacity=0.10,
                selected_color=PB_NAVY, selected_weight=2.2, selected_fill_opacity=0.18,
                tooltip_label="Iso ID: ",
            )

            if not g_show.empty:
                center, zoom = bounds_center_zoom(g_show)
                st.session_state["view_center"], st.session_state["view_zoom"] = center, zoom
                m.location, m.zoom_start = center, zoom

            click_gdf = g_show
            click_id_col = ISO_ID

        # -------------------
        # QUADRA (multi)
        # -------------------
        elif level == "quadra":
            iso_ids: Set[Any] = st.session_state["selected_iso_ids"]
            if not iso_ids:
                reset_to("isocrona")
                st.rerun()

            g_quadra = read_gdf_parquet(ensure_local_layer("quadra"))
            g_iso = read_gdf_parquet(ensure_local_layer("iso"))
            if g_quadra is None or g_quadra.empty:
                st.error("Quadras vazias/erro ao ler.")
                st.stop()
            g_quadra = _drop_bad_geoms(g_quadra)

            # sombra = isócronas selecionadas
            if g_iso is not None and not g_iso.empty:
                g_iso_sel = subset_by_parent_multi(_drop_bad_geoms(g_iso), ISO_ID, iso_ids)
                if g_iso_sel is not None and not g_iso_sel.empty:
                    add_shadow_parent(m, g_iso_sel, "Isócronas selecionadas")

            g_show = subset_by_parent_multi(g_quadra, QUADRA_PARENT, iso_ids)
            st.markdown("### Quadras (Isócronas selecionadas)")

            add_polygons_selectable(
                m, g_show, "Quadras (multi)", QUADRA_ID,
                selected_ids=st.session_state["selected_quadra_ids"],
                base_weight=0.45, fill_opacity=0.10,
                selected_color=PB_NAVY, selected_weight=2.0, selected_fill_opacity=0.20,
                tooltip_label="Quadra ID: ",
            )

            if not g_show.empty:
                center, zoom = bounds_center_zoom(g_show)
                st.session_state["view_center"], st.session_state["view_zoom"] = center, min(zoom + 1, 17)
                m.location, m.zoom_start = st.session_state["view_center"], st.session_state["view_zoom"]

            click_gdf = g_show
            click_id_col = QUADRA_ID

        # -------------------
        # DETAIL (Lotes + Setor censitário)
        # -------------------
        else:
            iso_ids: Set[Any] = st.session_state["selected_iso_ids"]
            quad_ids: Set[Any] = st.session_state["selected_quadra_ids"]
            if not iso_ids:
                reset_to("isocrona")
                st.rerun()

            st.markdown("### Detalhes (Lotes e Setor Censitário)")

            # sombra = quadras selecionadas (ou isócronas se não houver quadras)
            g_quadra = read_gdf_parquet(ensure_local_layer("quadra"))
            if g_quadra is not None and not g_quadra.empty and quad_ids:
                g_q_sel = subset_by_parent_multi(_drop_bad_geoms(g_quadra), QUADRA_ID, quad_ids)
                if g_q_sel is not None and not g_q_sel.empty:
                    add_shadow_parent(m, g_q_sel, "Quadras selecionadas (sombra)")
            else:
                g_iso = read_gdf_parquet(ensure_local_layer("iso"))
                if g_iso is not None and not g_iso.empty:
                    g_i_sel = subset_by_parent_multi(_drop_bad_geoms(g_iso), ISO_ID, iso_ids)
                    if g_i_sel is not None and not g_i_sel.empty:
                        add_shadow_parent(m, g_i_sel, "Isócronas selecionadas (sombra)")

            tabs = st.tabs(["🏠 Lotes", "📊 Setor censitário"])

            # --- LOTES
            with tabs[0]:
                if not quad_ids:
                    st.info("Selecione ao menos 1 quadra para carregar lotes.")
                else:
                    msg2 = _ensure_config(["lote"])
                    if msg2:
                        st.error(msg2)
                    else:
                        go_l = st.button("⬇️ Carregar LOTES (pode ser pesado)", type="primary")
                        if go_l:
                            g_lote = read_gdf_parquet(ensure_local_layer("lote"))
                            if g_lote is None or g_lote.empty:
                                st.error("Lotes vazios/erro ao ler.")
                            else:
                                g_lote = _drop_bad_geoms(g_lote)
                                if LOTE_PARENT not in g_lote.columns:
                                    st.error(f"Coluna pai ausente em Lotes: '{LOTE_PARENT}'")
                                else:
                                    g_show_l = subset_by_parent_multi(g_lote, LOTE_PARENT, quad_ids)
                                    if g_show_l.empty:
                                        st.warning("Nenhum lote encontrado para as quadras selecionadas.")
                                    else:
                                        add_polygons_selectable(
                                            m, g_show_l, "Lotes", LOTE_ID,
                                            selected_ids=set(),
                                            base_weight=0.25, fill_opacity=0.10,
                                            tooltip_label="Lote ID: ",
                                        )

            # --- CENSO
            with tabs[1]:
                msg2 = _ensure_config(["censo"])
                if msg2:
                    st.error(msg2)
                else:
                    go_c = st.button("⬇️ Carregar SETOR CENSITÁRIO", type="primary")
                    if go_c:
                        g_censo = read_gdf_parquet(ensure_local_layer("censo"))
                        if g_censo is None or g_censo.empty:
                            st.error("Setores vazios/erro ao ler.")
                        else:
                            g_censo = _drop_bad_geoms(g_censo)
                            if CENSO_PARENT not in g_censo.columns:
                                st.error(f"Coluna pai ausente em Setor censitário: '{CENSO_PARENT}'")
                            else:
                                g_show_c = subset_by_parent_multi(g_censo, CENSO_PARENT, iso_ids)
                                if g_show_c.empty:
                                    st.warning("Nenhum setor encontrado para as isócronas selecionadas.")
                                else:
                                    add_polygons_selectable(
                                        m, g_show_c, "Setor censitário", CENSO_ID,
                                        selected_ids=set(),
                                        base_weight=0.35, fill_opacity=0.08,
                                        tooltip_label="Censo ID: ",
                                    )

        # layer control
        try:
            folium.LayerControl(position="bottomright", collapsed=False).add_to(m)
        except Exception:
            pass

        # ✅ MUITO IMPORTANTE: pedir explicitamente o clique
        out = st_folium(
            m,
            height=780,
            use_container_width=True,
            key="map_view",
            returned_objects=["last_clicked"],
        )

        click = (out or {}).get("last_clicked")

        # -------------------
        # Click handler (somente onde faz sentido)
        # -------------------
        if click and click_gdf is not None and click_id_col is not None:
            picked = pick_feature_id(click_gdf, click, click_id_col)

            if picked is not None:
                if level == "subpref":
                    st.session_state["selected_subpref_id"] = picked
                    st.session_state["level"] = "distrito"
                    st.rerun()

                elif level == "distrito":
                    st.session_state["selected_distrito_id"] = picked
                    st.session_state["selected_iso_ids"] = set()
                    st.session_state["selected_quadra_ids"] = set()
                    st.session_state["level"] = "isocrona"
                    st.rerun()

                elif level == "isocrona":
                    _toggle_in_set("selected_iso_ids", picked)
                    st.session_state["selected_quadra_ids"] = set()
                    st.rerun()

                elif level == "quadra":
                    _toggle_in_set("selected_quadra_ids", picked)
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
