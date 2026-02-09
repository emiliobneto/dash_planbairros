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

import pandas as pd  # type: ignore

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
PARENT_FILL_OPACITY = 0.12
PARENT_STROKE_OPACITY = 0.30
PARENT_STROKE_WEIGHT = 0.7
PARENT_STROKE_DASH = "2,6"

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
# IDS DO ENCADEAMENTO
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

# colunas que precisam ser normalizadas por layer (ID + PAI)
LAYER_ID_COLS = {
    "subpref": [SUBPREF_ID],
    "dist": [DIST_ID, DIST_PARENT],
    "iso": [ISO_ID, ISO_PARENT],
    "quadra": [QUADRA_ID, QUADRA_PARENT],
    "lote": [LOTE_ID, LOTE_PARENT],
    "censo": [CENSO_ID, CENSO_PARENT],
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
# SECRETS + OVERRIDES (sidebar)
# =============================================================================
def _get_secret(key: str) -> str:
    try:
        return str(st.secrets.get(key, "")).strip()
    except Exception:
        return ""


# keys esperadas em secrets.toml:
# PB_SUBPREF_FILE_ID, PB_DISTRITO_FILE_ID, PB_ISOCRONAS_FILE_ID, PB_QUADRAS_FILE_ID, PB_LOTES_FILE_ID, PB_CENSO_FILE_ID
SECRETS_MAP = {
    "subpref":"https://drive.google.com/file/d/1vPY34cQLCoGfADpyOJjL9pNCYkVrmSZA/view?usp=drive_link",
    "dist":"https://drive.google.com/file/d/1K-t2BiSHN_D8De0oCFxzGdrEMhnGnh10/view?usp=drive_link",
    "iso":"https://drive.google.com/file/d/18ukyzMiYQ6vMqrU6-ctaPFbXMPX9XS9i/view?usp=drive_link",
    "quadra":"https://drive.google.com/file/d/1XKAYLNdt82ZPNAE-rseSuuaCFmjfn8IP/view?usp=drive_link",
    "lote":"https://drive.google.com/file/d/1oTFAZff1mVAWD6KQTJSz45I6B6pi6ceP/view?usp=drive_link",
    "censo":"https://drive.google.com/file/d/1APp7fxT2mgTpegVisVyQwjTRWOPz6Rgn/view?usp=drive_link",
}


def get_drive_raw(layer_key: str) -> str:
    """Prioridade: sidebar (session_state) > secrets.toml."""
    ui_key = f"drive_{layer_key}_raw"
    raw_ui = str(st.session_state.get(ui_key, "")).strip()
    if raw_ui:
        return raw_ui
    return _get_secret(SECRETS_MAP.get(layer_key, ""))


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
# ID NORMALIZATION (CORRIGE “LINKS” ENTRE ARQUIVOS)
# =============================================================================
def _id_to_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return str(v).strip()

    if isinstance(v, int):
        return str(v)

    s = str(v).strip()
    if s.endswith(".0"):
        core = s[:-2]
        if core.isdigit():
            return core
    return s


def normalize_id_cols(gdf: "gpd.GeoDataFrame", cols: Iterable[str]) -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty:
        return gdf
    g = gdf.copy()
    for c in cols:
        if c in g.columns:
            g[c] = g[c].map(_id_to_str)
    return g


# =============================================================================
# DRIVE helpers
# =============================================================================
_DRIVE_ID_RE_1 = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")
_DRIVE_ID_RE_2 = re.compile(r"[?&]id=([a-zA-Z0-9_-]+)")


def extract_drive_id(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", raw) and "http" not in raw.lower():
        return raw

    m = _DRIVE_ID_RE_1.search(raw)
    if m:
        return m.group(1)

    m = _DRIVE_ID_RE_2.search(raw)
    if m:
        return m.group(1)

    m = re.search(r"([a-zA-Z0-9_-]{20,})", raw)
    return m.group(1) if m else ""


def download_drive_file(file_id_or_url: str, dst: Path, label: str = "") -> Path:
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
    chunk = 1024 * 1024

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


def local_layer_path(layer_key: str) -> Path:
    return DATA_CACHE_DIR / LOCAL_FILENAMES[layer_key]


def layer_available_locally(layer_key: str) -> bool:
    p = local_layer_path(layer_key)
    return p.exists() and p.stat().st_size > 0


def ensure_local_layer(layer_key: str) -> Path:
    """Preferência: arquivo local -> Drive (se configurado)."""
    dst = local_layer_path(layer_key)
    if layer_available_locally(layer_key):
        return dst

    raw = get_drive_raw(layer_key)
    if not raw:
        raise RuntimeError(
            f"Layer '{layer_key}' não encontrada localmente em {dst.name} e não há FILE_ID/link configurado."
        )
    return download_drive_file(raw, dst, label=dst.name)


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

    pv = _id_to_str(parent_val)
    if pv is None:
        return child.iloc[0:0].copy()

    return child[child[parent_col].map(_id_to_str) == pv]


def subset_by_parent_multi(child: "gpd.GeoDataFrame", parent_col: str, parent_vals: Set[Any]) -> "gpd.GeoDataFrame":
    if child is None or child.empty:
        return child
    child = _drop_bad_geoms(child)
    if parent_col not in child.columns or not parent_vals:
        return child.iloc[0:0].copy()

    pset = {v for v in (_id_to_str(x) for x in parent_vals) if v is not None}
    if not pset:
        return child.iloc[0:0].copy()

    return child[child[parent_col].map(_id_to_str).isin(list(pset))]


def subset_by_id(gdf: "gpd.GeoDataFrame", id_col: str, id_val: Any) -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty:
        return gdf
    if id_col not in gdf.columns or id_val is None:
        return gdf.iloc[0:0].copy()

    iv = _id_to_str(id_val)
    if iv is None:
        return gdf.iloc[0:0].copy()

    return gdf[gdf[id_col].map(_id_to_str) == iv]


def subset_by_id_multi(gdf: "gpd.GeoDataFrame", id_col: str, ids: Set[Any]) -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty:
        return gdf
    if id_col not in gdf.columns or not ids:
        return gdf.iloc[0:0].copy()

    iset = {v for v in (_id_to_str(x) for x in ids) if v is not None}
    if not iset:
        return gdf.iloc[0:0].copy()

    return gdf[gdf[id_col].map(_id_to_str).isin(list(iset))]


# =============================================================================
# DIAGNÓSTICO FK
# =============================================================================
def diag_layer(gdf: "gpd.GeoDataFrame", title: str, cols: list[str]) -> None:
    with st.expander(f"🧪 Diagnóstico: {title}", expanded=False):
        if gdf is None or gdf.empty:
            st.warning("GDF vazio.")
            return
        st.write(f"Linhas: **{len(gdf):,}**")
        for c in cols:
            if c not in gdf.columns:
                st.error(f"Coluna ausente: {c}")
                continue
            s = gdf[c]
            n_null = int(s.isna().sum())
            n_unique = int(s.dropna().nunique())
            ex = s.dropna().astype(str).head(5).tolist()
            st.write(f"- **{c}** | dtype={s.dtype} | nulos={n_null} | únicos={n_unique} | ex={ex}")


def diag_fk(parent: "gpd.GeoDataFrame", parent_id: str, child: "gpd.GeoDataFrame", child_fk: str, label: str) -> None:
    with st.expander(f"🔗 Checagem FK: {label}", expanded=False):
        if parent is None or parent.empty or child is None or child.empty:
            st.warning("Pai ou filho vazio.")
            return
        if parent_id not in parent.columns or child_fk not in child.columns:
            st.error("Coluna ID/FK ausente.")
            return

        pvals = pd.Series(parent[parent_id].dropna().map(_id_to_str).unique())
        cvals = pd.Series(child[child_fk].dropna().map(_id_to_str).unique())

        pset = set(pvals.tolist())
        cset = set(cvals.tolist())

        missing = sorted(list(cset - pset))
        ok = len(cset) - len(missing)
        pct_ok = (ok / max(len(cset), 1)) * 100.0

        st.write(f"FK distintos no filho: **{len(cset):,}**")
        st.write(f"FK que existem no pai: **{ok:,}** (**{pct_ok:.2f}%**)")
        st.write(f"FK faltando no pai: **{len(missing):,}**")

        if missing:
            st.write("Exemplos (até 20):")
            st.code(", ".join(missing[:20]))


# =============================================================================
# CLICK HITTEST
# =============================================================================
def pick_feature_id(gdf: "gpd.GeoDataFrame", click_latlon: Dict[str, float], id_col: str) -> Optional[str]:
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
        return _id_to_str(hit.iloc[0][id_col])
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
    if folium is None or gdf is None or gdf.empty:
        return
    if id_col not in gdf.columns:
        return

    selected_ids = selected_ids or set()
    sel = {v for v in (_id_to_str(x) for x in selected_ids) if v is not None}

    tol = simplify_tol if simplify_tol is not None else 0.0006
    mini = gdf[[id_col, "geometry"]].copy()
    mini[id_col] = mini[id_col].map(_id_to_str)

    try:
        mini["geometry"] = mini.geometry.simplify(tol, preserve_topology=True)
    except Exception:
        pass
    mini = _drop_bad_geoms(mini)

    tooltip = _mk_tooltip(id_col, tooltip_prefix)

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

    if sel:
        sel_gdf = mini[mini[id_col].isin(list(sel))]
        if not sel_gdf.empty:
            fg_sel = folium.FeatureGroup(name=f"{name} (selecionados)", show=True)
            folium.GeoJson(
                data=sel_gdf.to_json(),
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
def data_sources_panel() -> None:
    st.markdown("<span class='pb-badge'>🗂️ Dados</span>", unsafe_allow_html=True)
    st.caption("Use arquivo local (data_cache) ou cole o link/ID do Drive aqui.")

    # mostra status local
    for k in ["subpref", "dist", "iso", "quadra", "lote", "censo"]:
        p = local_layer_path(k)
        ok = layer_available_locally(k)
        st.write(f"- `{p.name}`: {'✅ local' if ok else '—'}")

    st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)

    with st.expander("Configurar links/IDs do Drive", expanded=False):
        for k, secret_key in SECRETS_MAP.items():
            existing = str(st.session_state.get(f"drive_{k}_raw", "")).strip()
            hint = _get_secret(secret_key)
            placeholder = hint if hint else "Cole aqui o ID puro ou a URL do Drive"
            st.session_state[f"drive_{k}_raw"] = st.text_input(
                f"{k} ({LOCAL_FILENAMES[k]})",
                value=existing,
                placeholder=placeholder,
            )
        st.caption("Dica: se já estiver em secrets, você não precisa colar aqui.")

    st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='pb-note'>"
        "Se o app não encontrar o parquet local, ele tenta baixar do Drive usando o link/ID configurado. "
        "</div>",
        unsafe_allow_html=True,
    )


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
    data_sources_panel()


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
            try:
                sub_path = ensure_local_layer("subpref")
            except Exception as e:
                st.error(str(e))
                st.stop()

            g_sub = read_gdf_parquet(sub_path)
            if g_sub is None or g_sub.empty:
                st.error("Subprefeitura vazia/erro ao ler.")
                st.stop()

            g_sub = normalize_id_cols(_drop_bad_geoms(g_sub), LAYER_ID_COLS["subpref"])
            diag_layer(g_sub, "Subpref", [SUBPREF_ID])

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
            sp = _id_to_str(st.session_state["selected_subpref_id"])
            if sp is None:
                reset_to("subpref")
                st.rerun()

            try:
                dist_path = ensure_local_layer("dist")
                sub_path = ensure_local_layer("subpref")
            except Exception as e:
                st.error(str(e))
                st.stop()

            g_dist = read_gdf_parquet(dist_path)
            g_sub = read_gdf_parquet(sub_path)
            if g_dist is None or g_dist.empty:
                st.error("Distrito vazio/erro ao ler.")
                st.stop()
            if g_sub is None or g_sub.empty:
                st.error("Subprefeitura vazia/erro ao ler.")
                st.stop()

            g_dist = normalize_id_cols(_drop_bad_geoms(g_dist), LAYER_ID_COLS["dist"])
            g_sub = normalize_id_cols(_drop_bad_geoms(g_sub), LAYER_ID_COLS["subpref"])

            diag_layer(g_dist, "Distritos", [DIST_ID, DIST_PARENT])
            diag_fk(g_sub, SUBPREF_ID, g_dist, DIST_PARENT, "Distritos.subpref_id -> Subpref.subpref_id")

            g_parent = subset_by_id(g_sub, SUBPREF_ID, sp)
            add_parent_fill(m, g_parent, "Subpref selecionada (sombra)", simplify_tol=SIMPLIFY_TOL_BY_LEVEL["subpref"])

            g_show = subset_by_parent(g_dist, DIST_PARENT, sp)
            st.markdown(f"### Distritos (Subpref {sp})")
            add_polygons_selectable(
                m, g_show, "Distritos (clique)", DIST_ID,
                selected_ids=set(),
                base_weight=0.55, fill_opacity=0.05,
                selected_color=PB_NAVY,
                tooltip_prefix="Distrito: ",
                simplify_tol=SIMPLIFY_TOL_BY_LEVEL["distrito"],
            )

        # -------------------------
        # ISOCRONAS
        # -------------------------
        elif level == "isocrona":
            d = _id_to_str(st.session_state["selected_distrito_id"])
            if d is None:
                reset_to("distrito")
                st.rerun()

            try:
                iso_path = ensure_local_layer("iso")
                dist_path = ensure_local_layer("dist")
            except Exception as e:
                st.error(str(e))
                st.stop()

            g_iso = read_gdf_parquet(iso_path)
            g_dist = read_gdf_parquet(dist_path)
            if g_iso is None or g_iso.empty:
                st.error("Isócronas vazias/erro ao ler.")
                st.stop()
            if g_dist is None or g_dist.empty:
                st.error("Distritos vazios/erro ao ler.")
                st.stop()

            g_iso = normalize_id_cols(_drop_bad_geoms(g_iso), LAYER_ID_COLS["iso"])
            g_dist = normalize_id_cols(_drop_bad_geoms(g_dist), LAYER_ID_COLS["dist"])

            diag_layer(g_iso, "Isócronas", [ISO_ID, ISO_PARENT])
            diag_fk(g_dist, DIST_ID, g_iso, ISO_PARENT, "Isócronas.distrito_id -> Distritos.distrito_id")

            g_parent = subset_by_id(g_dist, DIST_ID, d)
            add_parent_fill(m, g_parent, "Distrito selecionado (sombra)", simplify_tol=SIMPLIFY_TOL_BY_LEVEL["distrito"])

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

        # -------------------------
        # QUADRAS
        # -------------------------
        elif level == "quadra":
            iso_ids = {v for v in (_id_to_str(x) for x in st.session_state["selected_iso_ids"]) if v is not None}
            if not iso_ids:
                reset_to("isocrona")
                st.rerun()

            try:
                quadra_path = ensure_local_layer("quadra")
                iso_path = ensure_local_layer("iso")
            except Exception as e:
                st.error(str(e))
                st.stop()

            g_quad = read_gdf_parquet(quadra_path)
            g_iso = read_gdf_parquet(iso_path)
            if g_quad is None or g_quad.empty:
                st.error("Quadras vazias/erro ao ler.")
                st.stop()
            if g_iso is None or g_iso.empty:
                st.error("Isócronas vazias/erro ao ler.")
                st.stop()

            g_quad = normalize_id_cols(_drop_bad_geoms(g_quad), LAYER_ID_COLS["quadra"])
            g_iso = normalize_id_cols(_drop_bad_geoms(g_iso), LAYER_ID_COLS["iso"])

            diag_layer(g_quad, "Quadras", [QUADRA_ID, QUADRA_PARENT])
            diag_fk(g_iso, ISO_ID, g_quad, QUADRA_PARENT, "Quadras.iso_id -> Isócronas.iso_id")

            g_parent = subset_by_id_multi(g_iso, ISO_ID, iso_ids)
            add_parent_fill(m, g_parent, "Isócronas selecionadas (sombra)", simplify_tol=SIMPLIFY_TOL_BY_LEVEL["isocrona"])

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

        # -------------------------
        # FINAL (lote/censo sob demanda)
        # -------------------------
        else:
            iso_ids = {v for v in (_id_to_str(x) for x in st.session_state["selected_iso_ids"]) if v is not None}
            quad_ids = {v for v in (_id_to_str(x) for x in st.session_state["selected_quadra_ids"]) if v is not None}

            mode = st.session_state["final_mode"]
            st.markdown("### Nível final")

            st.warning("Arquivos do nível final podem ser pesados. O download/leitura só ocorre ao clicar no botão abaixo.")
            go = st.button("⬇️ Carregar dados do nível final", type="primary")

            if go:
                if mode == "lote":
                    try:
                        lote_path = ensure_local_layer("lote")
                        quadra_path = ensure_local_layer("quadra")
                    except Exception as e:
                        st.error(str(e))
                        st.stop()

                    g_lote = read_gdf_parquet(lote_path)
                    g_quad = read_gdf_parquet(quadra_path)
                    if g_lote is None or g_lote.empty:
                        st.error("Lotes vazios/erro ao ler.")
                        st.stop()
                    if g_quad is None or g_quad.empty:
                        st.error("Quadras vazias/erro ao ler.")
                        st.stop()

                    g_lote = normalize_id_cols(_drop_bad_geoms(g_lote), LAYER_ID_COLS["lote"])
                    g_quad = normalize_id_cols(_drop_bad_geoms(g_quad), LAYER_ID_COLS["quadra"])

                    diag_layer(g_lote, "Lotes", [LOTE_ID, LOTE_PARENT])
                    diag_fk(g_quad, QUADRA_ID, g_lote, LOTE_PARENT, "Lotes.quadra_id -> Quadras.quadra_id")

                    if not quad_ids:
                        st.info("Selecione ao menos 1 quadra no nível anterior.")
                    else:
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
                    try:
                        censo_path = ensure_local_layer("censo")
                        iso_path = ensure_local_layer("iso")
                    except Exception as e:
                        st.error(str(e))
                        st.stop()

                    g_censo = read_gdf_parquet(censo_path)
                    g_iso = read_gdf_parquet(iso_path)
                    if g_censo is None or g_censo.empty:
                        st.error("Setores censitários vazios/erro ao ler.")
                        st.stop()
                    if g_iso is None or g_iso.empty:
                        st.error("Isócronas vazias/erro ao ler.")
                        st.stop()

                    g_censo = normalize_id_cols(_drop_bad_geoms(g_censo), LAYER_ID_COLS["censo"])
                    g_iso = normalize_id_cols(_drop_bad_geoms(g_iso), LAYER_ID_COLS["iso"])

                    diag_layer(g_censo, "Setor Censitário", [CENSO_ID, CENSO_PARENT])
                    diag_fk(g_iso, ISO_ID, g_censo, CENSO_PARENT, "Censo.iso_id -> Isócronas.iso_id")

                    if not iso_ids:
                        st.info("Selecione ao menos 1 isócrona no nível anterior.")
                    else:
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

        # layer control
        try:
            folium.LayerControl(position="bottomright", collapsed=False).add_to(m)
        except Exception:
            pass

        out = st_folium(
            m,
            height=780,
            use_container_width=True,
            key="map_view",
            returned_objects=["last_clicked"],
        )

        click = (out or {}).get("last_clicked")
        if click:
            if level == "subpref":
                sub_path = ensure_local_layer("subpref")
                g_sub = normalize_id_cols(_drop_bad_geoms(read_gdf_parquet(sub_path)), LAYER_ID_COLS["subpref"])
                picked = pick_feature_id(g_sub, click, SUBPREF_ID)
                if picked is not None:
                    st.session_state["selected_subpref_id"] = picked
                    st.session_state["level"] = "distrito"
                    st.rerun()

            elif level == "distrito":
                sp = _id_to_str(st.session_state["selected_subpref_id"])
                dist_path = ensure_local_layer("dist")
                g_dist = normalize_id_cols(_drop_bad_geoms(read_gdf_parquet(dist_path)), LAYER_ID_COLS["dist"])
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
                d = _id_to_str(st.session_state["selected_distrito_id"])
                iso_path = ensure_local_layer("iso")
                g_iso = normalize_id_cols(_drop_bad_geoms(read_gdf_parquet(iso_path)), LAYER_ID_COLS["iso"])
                g_show = subset_by_parent(g_iso, ISO_PARENT, d)
                picked = pick_feature_id(g_show, click, ISO_ID)
                if picked is not None:
                    _toggle_in_set("selected_iso_ids", picked)
                    st.rerun()

            elif level == "quadra":
                iso_ids = {v for v in (_id_to_str(x) for x in st.session_state["selected_iso_ids"]) if v is not None}
                quadra_path = ensure_local_layer("quadra")
                g_quad = normalize_id_cols(_drop_bad_geoms(read_gdf_parquet(quadra_path)), LAYER_ID_COLS["quadra"])
                g_show = subset_by_parent_multi(g_quad, QUADRA_PARENT, iso_ids)
                picked = pick_feature_id(g_show, click, QUADRA_ID)
                if picked is not None:
                    _toggle_in_set("selected_quadra_ids", picked)
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()



