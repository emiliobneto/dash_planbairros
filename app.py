# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Set, Iterable
import re
import base64

import streamlit as st

# geo
try:
    import geopandas as gpd  # type: ignore
    import folium  # type: ignore
    from folium.features import GeoJsonTooltip  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
    from shapely.geometry import Point  # type: ignore
except Exception:
    gpd = None  # type: ignore
    folium = None  # type: ignore
    GeoJsonTooltip = None  # type: ignore
    st_folium = None  # type: ignore
    Point = None  # type: ignore

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
    "marrom": "#C65534",  # telha como marrom
}
PB_NAVY = PB_COLORS["navy"]
PB_BROWN = PB_COLORS["telha"]

# Carto tiles explícito
CARTO_LIGHT_URL = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
CARTO_ATTR = "© OpenStreetMap contributors © CARTO"

# Suavização / acabamento
SMOOTH_FACTOR = 0.8
LINE_CAP = "round"
LINE_JOIN = "round"

# “Sombra” do nível acima (bem sutil, tracejada e fina)
PARENT_FILL_OPACITY = 0.12          # sombra suave
PARENT_STROKE_OPACITY = 0.30
PARENT_STROKE_WEIGHT = 0.7          # bem fino
PARENT_STROKE_DASH = "2,6"          # tracejado leve

# Simplificação por nível (somente para VISUAL, não para análise)
SIMPLIFY_TOL_BY_LEVEL = {
    "subpref": 0.0010,
    "distrito": 0.0008,
    "isocrona": 0.0006,
    "quadra": 0.00025,
    "lote": 0.00015,
    "censo": 0.00025,
}

# =============================================================================
# PATHS / ASSETS
# =============================================================================
REPO_ROOT = Path.cwd()
DATA_CACHE_DIR = REPO_ROOT / "data_cache"
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ASSETS_DIR = REPO_ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "logo_todos.jpg"
LOGO_HEIGHT = 46

# =============================================================================
# IDS DO ENCADEAMENTO (VOCÊ VAI GERAR)
# =============================================================================
SUBPREF_ID = "subpref_id"
DIST_ID = "distrito_id"
ISO_ID = "iso_id"
QUADRA_ID = "quadra_id"
LOTE_ID = "lote_id"
CENSO_ID = "censo_id"

DIST_PARENT = "subpref_id"     # distritos -> subpref
ISO_PARENT = "distrito_id"     # isócronas -> distrito
QUADRA_PARENT = "iso_id"       # quadras -> isócrona
LOTE_PARENT = "quadra_id"      # lotes -> quadra
CENSO_PARENT = "iso_id"        # setor censitário -> isócrona

LEVELS = ["subpref", "distrito", "isocrona", "quadra", "final"]

# =============================================================================
# DRIVE LINKS/IDS (via secrets.toml)
#  - você pode colocar: o ID puro "1abc..." OU a URL completa do drive
# =============================================================================
def _get_secret(key: str) -> str:
    try:
        return str(st.secrets.get(key, "")).strip()
    except Exception:
        return ""


# chaves esperadas em secrets.toml:
# PB_SUBPREF_FILE_ID, PB_DISTRITO_FILE_ID, PB_ISOCRONAS_FILE_ID, PB_QUADRAS_FILE_ID, PB_LOTES_FILE_ID, PB_CENSO_FILE_ID
GDRIVE_RAW = {
    "subpref": _get_secret("PB_SUBPREF_FILE_ID"),
    "dist": _get_secret("PB_DISTRITO_FILE_ID"),
    "iso": _get_secret("PB_ISOCRONAS_FILE_ID"),
    "quadra": _get_secret("PB_QUADRAS_FILE_ID"),
    "lote": _get_secret("PB_LOTES_FILE_ID"),
    "censo": _get_secret("PB_CENSO_FILE_ID"),
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
# DRIVE helpers
# =============================================================================
_DRIVE_ID_RE_1 = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")
_DRIVE_ID_RE_2 = re.compile(r"[?&]id=([a-zA-Z0-9_-]+)")


def extract_drive_id(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    # se já é um ID puro:
    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", raw) and "http" not in raw.lower():
        return raw

    m = _DRIVE_ID_RE_1.search(raw)
    if m:
        return m.group(1)

    m = _DRIVE_ID_RE_2.search(raw)
    if m:
        return m.group(1)

    # fallback: tenta pegar algo que pareça ID
    m = re.search(r"([a-zA-Z0-9_-]{20,})", raw)
    return m.group(1) if m else ""


def _ensure_file_ids_configured(keys: list[str]) -> Optional[str]:
    missing = []
    for k in keys:
        if not extract_drive_id(GDRIVE_RAW.get(k, "")):
            missing.append(k)
    if missing:
        return (
            "Faltam FILE_IDs (ou links válidos) no secrets.toml para: "
            + ", ".join(missing)
            + "."
        )
    return None


def download_drive_file(file_id_or_url: str, dst: Path, label: str = "") -> Path:
    """Download robusto (streaming) via requests. Evita manter o conteúdo em memória."""
    import requests

    file_id = extract_drive_id(file_id_or_url)
    if not file_id:
        raise RuntimeError("FILE_ID inválido (não foi possível extrair ID do link).")

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

    ui_label = label or dst.name
    prog = st.progress(0, text=f"Baixando {ui_label}…")
    downloaded = 0

    with open(dst, "wb") as f:
        for part in response.iter_content(chunk_size=chunk):
            if not part:
                continue
            f.write(part)
            downloaded += len(part)
            if total > 0:
                pct = min(int(downloaded * 100 / total), 100)
                prog.progress(pct, text=f"Baixando {ui_label}… {pct}%")

    prog.empty()
    return dst


def ensure_local_layer(layer_key: str) -> Path:
    raw = GDRIVE_RAW.get(layer_key, "")
    if not raw:
        raise RuntimeError(f"FILE_ID não configurado para layer '{layer_key}'.")
    dst = DATA_CACHE_DIR / LOCAL_FILENAMES[layer_key]
    return download_drive_file(raw, dst, label=LOCAL_FILENAMES[layer_key])


# =============================================================================
# READ / FILTER (cache_data)
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


def simplify_for_display(gdf: "gpd.GeoDataFrame", tol: float) -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty:
        return gdf
    g = gdf.copy()
    try:
        g["geometry"] = g.geometry.simplify(tol, preserve_topology=True)
    except Exception:
        pass
    return _drop_bad_geoms(g)


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


def subset_by_id(gdf: "gpd.GeoDataFrame", id_col: str, id_val: Any) -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty:
        return gdf
    if id_col not in gdf.columns or id_val is None:
        return gdf.iloc[0:0].copy()
    return gdf[gdf[id_col] == id_val]


def subset_by_id_multi(gdf: "gpd.GeoDataFrame", id_col: str, ids: Set[Any]) -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty:
        return gdf
    if id_col not in gdf.columns or not ids:
        return gdf.iloc[0:0].copy()
    return gdf[gdf[id_col].isin(list(ids))]


# =============================================================================
# CLICK HITTEST
# =============================================================================
def pick_feature_id(gdf: "gpd.GeoDataFrame", click_latlon: Dict[str, float], id_col: str) -> Optional[Any]:
    """Retorna o id_col da feição que contém/intersecta o clique."""
    if gdf is None or gdf.empty or not click_latlon:
        return None
    if id_col not in gdf.columns:
        return None
    if Point is None:
        return None

    lat = click_latlon.get("lat")
    lng = click_latlon.get("lng")
    if lat is None or lng is None:
        return None

    try:
        pt = Point(lng, lat)
        cand = gdf

        # usa sindex se existir (bem mais rápido)
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

    # panes: sombra (pai) abaixo; shapes; linhas por cima
    try:
        folium.map.CustomPane("parent_fill", z_index=610).add_to(m)
        folium.map.CustomPane("detail_shapes", z_index=640).add_to(m)
        folium.map.CustomPane("top_lines", z_index=670).add_to(m)
    except Exception:
        pass

    return m


def _mk_tooltip(id_col: str, prefix: str) -> Optional[Any]:
    if GeoJsonTooltip is None:
        return None
    return GeoJsonTooltip(
        fields=[id_col],
        aliases=[prefix],
        sticky=True,
        labels=True,
        localize=True,
        max_width=320,
    )


def add_outline(
    m,
    gdf: "gpd.GeoDataFrame",
    name: str,
    color="#111111",
    weight=1.2,
    show=True,
    simplify_tol: float = 0.0006,
) -> None:
    """Contorno com acabamento melhor (cap/join redondos)."""
    if folium is None or gdf is None or gdf.empty:
        return

    line = gdf[["geometry"]].copy()
    try:
        line["geometry"] = line.geometry.boundary
        line["geometry"] = line.geometry.simplify(simplify_tol, preserve_topology=True)
    except Exception:
        return
    line = _drop_bad_geoms(line)

    fg = folium.FeatureGroup(name=name, show=show)
    folium.GeoJson(
        data=line.to_json(),
        pane="top_lines",
        smooth_factor=SMOOTH_FACTOR,
        style_function=lambda _f: {
            "fillOpacity": 0,
            "color": color,
            "weight": weight,
            "opacity": 0.95,
            "lineCap": LINE_CAP,
            "lineJoin": LINE_JOIN,
        },
    ).add_to(fg)
    fg.add_to(m)


def add_parent_fill(
    m,
    gdf: "gpd.GeoDataFrame",
    name: str,
    pane: str = "parent_fill",
    fill_color: str = PB_BROWN,
    fill_opacity: float = PARENT_FILL_OPACITY,
    stroke_color: str = PB_BROWN,
    stroke_weight: float = PARENT_STROKE_WEIGHT,
    stroke_opacity: float = PARENT_STROKE_OPACITY,
    dash_array: str = PARENT_STROKE_DASH,
    simplify_tol: float = 0.0006,
) -> None:
    """Camada do nível acima como 'sombra' (fill sutil + borda tracejada fina)."""
    if folium is None or gdf is None or gdf.empty:
        return

    g = gdf[["geometry"]].copy()
    try:
        g["geometry"] = g.geometry.simplify(simplify_tol, preserve_topology=True)
    except Exception:
        pass
    g = _drop_bad_geoms(g)

    fg = folium.FeatureGroup(name=name, show=True)
    folium.GeoJson(
        data=g.to_json(),
        pane=pane,
        smooth_factor=SMOOTH_FACTOR,
        style_function=lambda _f: {
            "color": stroke_color,
            "weight": stroke_weight,
            "opacity": stroke_opacity,
            "dashArray": dash_array,
            "lineCap": LINE_CAP,
            "lineJoin": LINE_JOIN,
            "fillColor": fill_color,
            "fillOpacity": fill_opacity,
        },
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
    tooltip_prefix: str = "ID: ",
    simplify_tol: Optional[float] = None,
) -> None:
    """
    Seleção mais rápida:
      - camada base com estilo fixo
      - overlay só com selecionados (pequeno)
    """
    if folium is None or gdf is None or gdf.empty:
        return
    if id_col not in gdf.columns:
        return

    selected_ids = selected_ids or set()

    tol = simplify_tol if simplify_tol is not None else 0.0006
    mini = gdf[[id_col, "geometry"]].copy()
    try:
        mini["geometry"] = mini.geometry.simplify(tol, preserve_topology=True)
    except Exception:
        pass
    mini = _drop_bad_geoms(mini)

    tooltip = _mk_tooltip(id_col, tooltip_prefix)

    # 1) Base (fixa)
    fg_base = folium.FeatureGroup(name=name, show=True)
    folium.GeoJson(
        data=mini.to_json(),
        pane=pane,
        smooth_factor=SMOOTH_FACTOR,
        style_function=lambda _f: {
            "color": base_color,
            "weight": base_weight,
            "opacity": 0.9,
            "lineCap": LINE_CAP,
            "lineJoin": LINE_JOIN,
            "fillColor": fill_color,
            "fillOpacity": fill_opacity,
        },
        highlight_function=lambda _f: {
            "weight": base_weight + 1.2,
            "fillOpacity": min(fill_opacity + 0.10, 0.35),
        },
        tooltip=tooltip,
    ).add_to(fg_base)
    fg_base.add_to(m)

    # 2) Selecionados (overlay pequeno)
    if selected_ids:
        sel = mini[mini[id_col].isin(list(selected_ids))]
        if not sel.empty:
            fg_sel = folium.FeatureGroup(name=f"{name} (selecionados)", show=True)
            folium.GeoJson(
                data=sel.to_json(),
                pane=pane,
                smooth_factor=SMOOTH_FACTOR,
                style_function=lambda _f: {
                    "color": selected_color,
                    "weight": selected_weight,
                    "opacity": 0.95,
                    "lineCap": LINE_CAP,
                    "lineJoin": LINE_JOIN,
                    "fillColor": fill_color,
                    "fillOpacity": selected_fill_opacity,
                },
                tooltip=tooltip,
            ).add_to(fg_sel)
            fg_sel.add_to(m)


# =============================================================================
# UI
# =============================================================================
def left_panel() -> None:
    st.markdown("<span class='pb-badge'>🧭 Fluxo</span>", unsafe_allow_html=True)
    st.caption("Subprefeitura → Distrito → Isócronas → Quadras → (Lotes | Setor Censitário)")

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

    if st.session_state["level"] == "isocrona":
        st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)
        ok = len(st.session_state["selected_iso_ids"]) > 0
        st.button(
            "➡️ Ir para Quadras",
            use_container_width=True,
            disabled=not ok,
            on_click=lambda: st.session_state.update({"level": "quadra"}),
        )

    if st.session_state["level"] == "quadra":
        st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)
        ok = len(st.session_state["selected_quadra_ids"]) > 0
        st.button(
            "➡️ Ir para Nível Final",
            use_container_width=True,
            disabled=not ok,
            on_click=lambda: st.session_state.update({"level": "final"}),
        )

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
    st.markdown("<span class='pb-badge'>⚙️ Performance</span>", unsafe_allow_html=True)
    st.markdown(
        "<div class='pb-note'>"
        "Para evitar crash: os arquivos são baixados e lidos somente quando o nível precisa. "
        "O nível final (lotes/setor) só carrega ao clicar no botão de download."
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
        st.markdown(
            f"**Iso: {len(st.session_state['selected_iso_ids'])} | "
            f"Quad: {len(st.session_state['selected_quadra_ids'])}**"
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

    # IDs mínimos (não exige lote/censo no boot)
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

        # -------------------------
        # SUBPREF
        # -------------------------
        if level == "subpref":
            st.markdown("### Subprefeituras")
            sub_path = ensure_local_layer("subpref")
            g_sub = read_gdf_parquet(sub_path)
            if g_sub is None or g_sub.empty:
                st.error("Subprefeitura vazia/erro ao ler.")
                st.stop()
            g_sub = _drop_bad_geoms(g_sub)

            if SUBPREF_ID not in g_sub.columns:
                st.error(f"Coluna obrigatória ausente em Subprefeitura: '{SUBPREF_ID}'.")
                st.stop()

            add_outline(
                m, g_sub, "Subprefeituras (linha)",
                color="#111111", weight=1.25,
                simplify_tol=SIMPLIFY_TOL_BY_LEVEL["subpref"],
            )
            add_polygons_selectable(
                m, g_sub, "Subprefeituras (clique)", SUBPREF_ID,
                selected_ids=set(),
                base_weight=0.8, fill_opacity=0.03,
                selected_color=PB_NAVY,
                tooltip_prefix="Subpref: ",
                simplify_tol=SIMPLIFY_TOL_BY_LEVEL["subpref"],
            )

        # -------------------------
        # DISTRITO
        # -------------------------
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

            if DIST_PARENT not in g_dist.columns or DIST_ID not in g_dist.columns:
                st.error(f"Colunas obrigatórias ausentes em Distritos: '{DIST_PARENT}' e/ou '{DIST_ID}'.")
                st.stop()

            # sombra: subpref selecionada
            sub_path = ensure_local_layer("subpref")
            g_sub = _drop_bad_geoms(read_gdf_parquet(sub_path))
            g_parent = subset_by_id(g_sub, SUBPREF_ID, sp)
            add_parent_fill(
                m, g_parent, "Subpref selecionada (sombra)",
                simplify_tol=SIMPLIFY_TOL_BY_LEVEL["subpref"],
            )

            g_show = subset_by_parent(g_dist, DIST_PARENT, sp)
            st.markdown(f"### Distritos (Subpref {sp})")

            add_outline(
                m, g_show, "Distritos (linha)",
                color="#111111", weight=1.05,
                simplify_tol=SIMPLIFY_TOL_BY_LEVEL["distrito"],
            )
            add_polygons_selectable(
                m, g_show, "Distritos (clique)", DIST_ID,
                selected_ids=set(),
                base_weight=0.55, fill_opacity=0.05,
                selected_color=PB_NAVY,
                tooltip_prefix="Distrito: ",
                simplify_tol=SIMPLIFY_TOL_BY_LEVEL["distrito"],
            )

            if not g_show.empty:
                center, zoom = bounds_center_zoom(g_show)
                st.session_state["view_center"], st.session_state["view_zoom"] = center, zoom
                m.location, m.zoom_start = center, zoom

        # -------------------------
        # ISOCRONAS (multi)
        # -------------------------
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

            if ISO_PARENT not in g_iso.columns or ISO_ID not in g_iso.columns:
                st.error(f"Colunas obrigatórias ausentes em Isócronas: '{ISO_PARENT}' e/ou '{ISO_ID}'.")
                st.stop()

            # sombra: distrito selecionado
            dist_path = ensure_local_layer("dist")
            g_dist = _drop_bad_geoms(read_gdf_parquet(dist_path))
            g_parent = subset_by_id(g_dist, DIST_ID, d)
            add_parent_fill(
                m, g_parent, "Distrito selecionado (sombra)",
                simplify_tol=SIMPLIFY_TOL_BY_LEVEL["distrito"],
            )

            g_show = subset_by_parent(g_iso, ISO_PARENT, d)
            st.markdown(f"### Isócronas (Distrito {d}) — clique para selecionar (multi)")

            add_polygons_selectable(
                m, g_show, "Isócronas", ISO_ID,
                selected_ids=st.session_state["selected_iso_ids"],
                base_weight=0.55, fill_opacity=0.10,
                selected_color=PB_NAVY, selected_weight=2.2, selected_fill_opacity=0.22,
                tooltip_prefix="Isócrona: ",
                simplify_tol=SIMPLIFY_TOL_BY_LEVEL["isocrona"],
            )

            if not g_show.empty:
                center, zoom = bounds_center_zoom(g_show)
                st.session_state["view_center"], st.session_state["view_zoom"] = center, zoom
                m.location, m.zoom_start = center, zoom

        # -------------------------
        # QUADRAS (multi)
        # -------------------------
        elif level == "quadra":
            iso_ids: Set[Any] = st.session_state["selected_iso_ids"]
            if not iso_ids:
                reset_to("isocrona")
                st.rerun()

            quadra_path = ensure_local_layer("quadra")
            g_quad = read_gdf_parquet(quadra_path)
            if g_quad is None or g_quad.empty:
                st.error("Quadras vazias/erro ao ler.")
                st.stop()
            g_quad = _drop_bad_geoms(g_quad)

            if QUADRA_PARENT not in g_quad.columns or QUADRA_ID not in g_quad.columns:
                st.error(f"Colunas obrigatórias ausentes em Quadras: '{QUADRA_PARENT}' e/ou '{QUADRA_ID}'.")
                st.stop()

            # sombra: isócronas selecionadas
            iso_path = ensure_local_layer("iso")
            g_iso = _drop_bad_geoms(read_gdf_parquet(iso_path))
            g_parent = subset_by_id_multi(g_iso, ISO_ID, iso_ids)
            add_parent_fill(
                m, g_parent, "Isócronas selecionadas (sombra)",
                simplify_tol=SIMPLIFY_TOL_BY_LEVEL["isocrona"],
            )

            g_show = subset_by_parent_multi(g_quad, QUADRA_PARENT, iso_ids)
            st.markdown("### Quadras (Isócronas selecionadas) — clique para selecionar (multi)")

            add_polygons_selectable(
                m, g_show, "Quadras", QUADRA_ID,
                selected_ids=st.session_state["selected_quadra_ids"],
                base_weight=0.40, fill_opacity=0.10,
                selected_color=PB_NAVY, selected_weight=2.0, selected_fill_opacity=0.22,
                tooltip_prefix="Quadra: ",
                simplify_tol=SIMPLIFY_TOL_BY_LEVEL["quadra"],
            )

            if not g_show.empty:
                center, zoom = bounds_center_zoom(g_show)
                st.session_state["view_center"], st.session_state["view_zoom"] = center, min(zoom + 1, 17)
                m.location, m.zoom_start = st.session_state["view_center"], st.session_state["view_zoom"]

        # -------------------------
        # FINAL (lotes OU setor censitário) — download sob demanda
        # -------------------------
        else:
            iso_ids: Set[Any] = st.session_state["selected_iso_ids"]
            quad_ids: Set[Any] = st.session_state["selected_quadra_ids"]

            mode = st.session_state["final_mode"]
            st.markdown("### Nível final")

            if mode == "lote" and not quad_ids:
                st.info("Selecione ao menos 1 quadra para visualizar lotes.")
                reset_to("quadra")
                st.rerun()

            if not iso_ids:
                st.info("Selecione ao menos 1 isócrona.")
                reset_to("isocrona")
                st.rerun()

            # sombra: mantém o nível acima bem sutil
            if mode == "lote":
                quadra_path = ensure_local_layer("quadra")
                g_quad = _drop_bad_geoms(read_gdf_parquet(quadra_path))
                g_parent = subset_by_id_multi(g_quad, QUADRA_ID, quad_ids)
                add_parent_fill(
                    m, g_parent, "Quadras selecionadas (sombra)",
                    simplify_tol=SIMPLIFY_TOL_BY_LEVEL["quadra"],
                )
            else:
                iso_path = ensure_local_layer("iso")
                g_iso = _drop_bad_geoms(read_gdf_parquet(iso_path))
                g_parent = subset_by_id_multi(g_iso, ISO_ID, iso_ids)
                add_parent_fill(
                    m, g_parent, "Isócronas selecionadas (sombra)",
                    simplify_tol=SIMPLIFY_TOL_BY_LEVEL["isocrona"],
                )

            # botão explícito para evitar crash em rerun automático
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

                    if LOTE_PARENT not in g_lote.columns or LOTE_ID not in g_lote.columns:
                        st.error(f"Colunas obrigatórias ausentes em Lotes: '{LOTE_PARENT}' e/ou '{LOTE_ID}'.")
                        st.stop()

                    g_show = subset_by_parent_multi(g_lote, LOTE_PARENT, quad_ids)

                    st.markdown("#### Lotes (Quadras selecionadas)")
                    add_polygons_selectable(
                        m, g_show, "Lotes", LOTE_ID,
                        selected_ids=set(),
                        base_weight=0.22, fill_opacity=0.10,
                        selected_color=PB_NAVY,
                        tooltip_prefix="Lote: ",
                        simplify_tol=SIMPLIFY_TOL_BY_LEVEL["lote"],
                    )

                else:
                    msg2 = _ensure_file_ids_configured(["censo"])
                    if msg2:
                        st.error(msg2)
                        st.stop()

                    censo_path = ensure_local_layer("censo")
                    g_censo = read_gdf_parquet(censo_path)
                    if g_censo is None or g_censo.empty:
                        st.error("Setores censitários vazios/erro ao ler.")
                        st.stop()
                    g_censo = _drop_bad_geoms(g_censo)

                    if CENSO_PARENT not in g_censo.columns or CENSO_ID not in g_censo.columns:
                        st.error(f"Colunas obrigatórias ausentes em Setor censitário: '{CENSO_PARENT}' e/ou '{CENSO_ID}'.")
                        st.stop()

                    g_show = subset_by_parent_multi(g_censo, CENSO_PARENT, iso_ids)

                    st.markdown("#### Setores Censitários (Isócronas selecionadas)")
                    add_polygons_selectable(
                        m, g_show, "Setor censitário", CENSO_ID,
                        selected_ids=set(),
                        base_weight=0.35, fill_opacity=0.08,
                        selected_color=PB_NAVY,
                        tooltip_prefix="Setor: ",
                        simplify_tol=SIMPLIFY_TOL_BY_LEVEL["censo"],
                    )

                if g_show is not None and not g_show.empty:
                    center, zoom = bounds_center_zoom(g_show)
                    st.session_state["view_center"], st.session_state["view_zoom"] = center, min(zoom + 1, 18)
                    m.location, m.zoom_start = st.session_state["view_center"], st.session_state["view_zoom"]

        # layer control
        try:
            folium.LayerControl(position="bottomright", collapsed=False).add_to(m)
        except Exception:
            pass

        out = st_folium(m, height=780, use_container_width=True, key="map_view", returned_objects=[])

        # -------------------------
        # CLICK ACTIONS
        # -------------------------
        click = (out or {}).get("last_clicked")
        if click:
            if level == "subpref":
                sub_path = ensure_local_layer("subpref")
                g_sub = _drop_bad_geoms(read_gdf_parquet(sub_path))
                picked = pick_feature_id(g_sub, click, SUBPREF_ID)
                if picked is not None:
                    st.session_state["selected_subpref_id"] = picked
                    st.session_state["level"] = "distrito"
                    st.rerun()

            elif level == "distrito":
                sp = st.session_state["selected_subpref_id"]
                dist_path = ensure_local_layer("dist")
                g_dist = _drop_bad_geoms(read_gdf_parquet(dist_path))
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
                g_iso = _drop_bad_geoms(read_gdf_parquet(iso_path))
                g_show = subset_by_parent(g_iso, ISO_PARENT, d)
                picked = pick_feature_id(g_show, click, ISO_ID)
                if picked is not None:
                    _toggle_in_set("selected_iso_ids", picked)
                    # permanece em isocrona (multi mais fluida)
                    st.rerun()

            elif level == "quadra":
                iso_ids = st.session_state["selected_iso_ids"]
                quadra_path = ensure_local_layer("quadra")
                g_quad = _drop_bad_geoms(read_gdf_parquet(quadra_path))
                g_show = subset_by_parent_multi(g_quad, QUADRA_PARENT, iso_ids)
                picked = pick_feature_id(g_show, click, QUADRA_ID)
                if picked is not None:
                    _toggle_in_set("selected_quadra_ids", picked)
                    # permanece em quadra (multi mais fluida)
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
