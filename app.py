# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import base64
import re

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
    "marrom": "#C65534",
}
PB_NAVY = PB_COLORS["navy"]
PB_BROWN = PB_COLORS["telha"]
PB_BTN = "#1C6880"

CARTO_LIGHT_URL = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
CARTO_ATTR = "© OpenStreetMap contributors © CARTO"

SMOOTH_FACTOR = 1.0
LINE_CAP = "round"
LINE_JOIN = "round"

PARENT_FILL_OPACITY = 0.16
PARENT_STROKE_OPACITY = 0.35
PARENT_STROKE_WEIGHT = 0.7
PARENT_STROKE_DASH = "2,6"

SIMPLIFY_TOL_BY_LEVEL = {
    "subpref": 0.0012,
    "distrito": 0.0008,
    "isocrona": 0.0006,
    "censo": 0.00035,
    "quadra": 0.00025,
    "lote": 0.00012,
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
# VISUALIZAÇÕES (Clusters / Isócronas)
# =============================================================================
QUADRAS_CSV_FILENAME = "quadras.csv"
QUADRAS_CSV_SECRET_KEY = "PB_QUADRAS_CSV_FILE_ID"
QUADRAS_CSV_FALLBACK_URL = (
    "https://drive.google.com/file/d/1_WKryQlu_jZL1xsgAmQDrI81aSdzKYsc/view?usp=drive_link"
)
CLUSTER_COL = "Cluster"
ISO_CLASS_COL = "nova_class"

CLUSTER_COLOR_MAP = {
    0: "#bf7db2",
    1: "#f7bd6a",
    2: "#cf651f",
    3: "#cf651f",
    4: "#793393",
}
CLUSTER_LABEL_MAP = {
    0: "Alta Densidade Periférica",
    1: "Uso Misto Intermediário",
    2: "Média Densidade Periférica",
    3: "Uso Misto Verticalizado Central",
    4: "Predominância Comercial e de Serviços",
}
CLUSTER_NULL_COLOR = "#c8c8c8"

ISO_TRANSITION_SET = {1, 3, 6}
ISO_TRANSITION_LABEL = "Área de transição"
ISO_TRANSITION_COLOR = "#7f6a5c"

ISO_VALUE_TO_CLASSNUM = {0: 1, 2: 2, 4: 3, 5: 4, 7: 5, 8: 6, 9: 7}
ISO_CLASSNUM_TO_COLOR = {
    1: "#f7f7f7",
    2: "#d8daeb",
    3: "#8073ac",
    4: "#b2abd2",
    5: "#b35806",
    6: "#e08214",
    7: "#542788",
}
ISO_DEFAULT_COLOR = "#ffffff"

# =============================================================================
# IDS / CHAVES
# =============================================================================
SUBPREF_ID = "subpref_id"
DIST_ID = "distrito_id"
ISO_ID = "iso_id"
QUADRA_ID = "quadra_id"
QUADRA_UID = "quadra_uid"  # iso_id__quadra_id
LOTE_ID = "lote_id"
CENSO_ID = "censo_id"

DIST_PARENT = SUBPREF_ID
ISO_PARENT = DIST_ID

QUADRA_PARENTS = {
    "iso": ISO_ID,
    "censo": CENSO_ID,
}

LOTE_PARENT = QUADRA_ID
CENSO_PARENT = ISO_ID

LEVELS = ["subpref", "distrito", "isocrona", "censo", "quadra", "final"]

LAYER_ID_COLS = {
    "subpref": [SUBPREF_ID],
    "dist": [DIST_ID, DIST_PARENT],
    "iso": [ISO_ID, ISO_PARENT],
    "quadra": [QUADRA_ID, ISO_ID, CENSO_ID, QUADRA_UID],
    "lote": [LOTE_ID, LOTE_PARENT, QUADRA_ID, ISO_ID],
    "censo": [CENSO_ID, CENSO_PARENT, QUADRA_ID, ISO_ID],
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
# NORMALIZAÇÃO DE COLUNAS / IDS
# =============================================================================
def _mk_aliases(base: str) -> Set[str]:
    b = base.strip()
    return {
        b,
        b.upper(),
        b.title(),
        b.replace("_", ""),
        b.replace("_", "").upper(),
        f"{b} ",
        f" {b}",
    }


COL_ALIASES: Dict[str, Set[str]] = {
    SUBPREF_ID: _mk_aliases(SUBPREF_ID),
    DIST_ID: _mk_aliases(DIST_ID),
    ISO_ID: _mk_aliases(ISO_ID),
    QUADRA_ID: _mk_aliases(QUADRA_ID),
    LOTE_ID: _mk_aliases(LOTE_ID),
    # ✅ ampliar aliases comuns e variações para setor
    CENSO_ID: _mk_aliases(CENSO_ID) | {"cendo_id", "CENDO_ID", "setor_id", "id_setor", "codigo_setor", "cd_setor"},
}


def standardize_columns(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """
    Padroniza nomes de colunas:
    - strip
    - colapsa múltiplos espaços
    - resolve aliases para nomes canônicos
    """
    if gdf is None or gdf.empty:
        return gdf

    ren: Dict[str, str] = {}
    for c in gdf.columns:
        raw = str(c)
        cl = raw.strip()
        cl = re.sub(r"\s+", " ", cl)  # colapsa espaços internos
        low = cl.lower()

        if raw != cl:
            ren[raw] = cl

        for canon, aliases in COL_ALIASES.items():
            aliases_norm = {a.strip() for a in aliases}
            aliases_low = {a.strip().lower() for a in aliases}
            if cl in aliases_norm or low in aliases_low:
                ren[cl] = canon

    return gdf.rename(columns=ren)


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


def normalize_quadra_id(v: Any, width: int = 6) -> Optional[str]:
    s = _id_to_str(v)
    if s is None:
        return None
    return s.zfill(width) if s.isdigit() else s


def make_quadra_uid(iso_id: Any, quadra_id: Any) -> Optional[str]:
    iso = _id_to_str(iso_id)
    qid = _id_to_str(quadra_id)
    if not iso or not qid:
        return None
    return f"{iso}__{qid}"


def split_quadra_uid(uid: str) -> Tuple[Optional[str], Optional[str]]:
    if not uid or "__" not in uid:
        return (None, None)
    a, b = uid.split("__", 1)
    return (_id_to_str(a), _id_to_str(b))


def normalize_id_cols(gdf: "gpd.GeoDataFrame", cols: Iterable[str]) -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty:
        return gdf
    g = gdf.copy()
    for c in cols:
        if c in g.columns:
            g[c] = g[c].map(_id_to_str)
    return g


# =============================================================================
# HEADER / CSS
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
            padding:12px;
        }}
        button[data-testid="stBaseButton-primary"],
        div[data-testid="stBaseButton-primary"] > button {{
            background:{PB_BTN} !important;
            color:#fff !important;
            border:1px solid {PB_BTN} !important;
        }}
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
MAP_KEY = "map_view"


def init_state() -> None:
    st.session_state.setdefault("level", "subpref")
    st.session_state.setdefault("last_level", None)

    st.session_state.setdefault("selected_subpref_id", None)
    st.session_state.setdefault("selected_distrito_id", None)

    st.session_state.setdefault("selected_iso_ids", set())
    st.session_state.setdefault("selected_censo_ids", set())
    st.session_state.setdefault("selected_quadra_ids", set())

    st.session_state.setdefault("iso_next_mode", "quadra")  # "quadra" | "censo"

    # filtro usado para transição censo -> quadra
    st.session_state.setdefault("quadra_filter_col", CENSO_ID)
    st.session_state.setdefault("quadra_filter_uids", set())

    st.session_state.setdefault("view_center", (-23.55, -46.63))
    st.session_state.setdefault("view_zoom", 11)

    st.session_state.setdefault("last_click_sig", "")

    st.session_state.setdefault("_geojson_cache", {})
    st.session_state.setdefault("_geojson_cache_order", [])
    st.session_state.setdefault("_layer_cache", {})
    st.session_state.setdefault("_layer_cache_meta", {})

    st.session_state.setdefault("_ui_action_sig", 0)
    st.session_state.setdefault("_ui_action_sig_seen", 0)

    st.session_state.setdefault("_map_level_rendered", None)
    st.session_state.setdefault("_quadra_id_col_map", QUADRA_UID)

    st.session_state.setdefault("variable", None)

    st.session_state.setdefault("final_load_sig", "")
    st.session_state.setdefault("final_loaded", False)
    st.session_state.setdefault("_final_lotes_gdf", None)

    # ✅ debug (para confirmar colunas reais carregadas)
    st.session_state.setdefault("debug_schema", False)


def mark_ui_action() -> None:
    st.session_state["_ui_action_sig"] = int(st.session_state.get("_ui_action_sig", 0)) + 1


def _geojson_cache_reset() -> None:
    st.session_state["_geojson_cache"] = {}
    st.session_state["_geojson_cache_order"] = []


def _final_reset() -> None:
    st.session_state["final_load_sig"] = ""
    st.session_state["final_loaded"] = False
    st.session_state["_final_lotes_gdf"] = None


def reset_to(level: str, *, clear_click_sig: bool = True) -> None:
    st.session_state["level"] = level
    if clear_click_sig:
        st.session_state["last_click_sig"] = ""
    _geojson_cache_reset()
    _final_reset()

    if level == "subpref":
        st.session_state["selected_subpref_id"] = None
        st.session_state["selected_distrito_id"] = None
        st.session_state["selected_iso_ids"] = set()
        st.session_state["selected_censo_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["iso_next_mode"] = "quadra"
        st.session_state["quadra_filter_col"] = CENSO_ID
        st.session_state["quadra_filter_uids"] = set()
        st.session_state["view_center"] = (-23.55, -46.63)
        st.session_state["view_zoom"] = 11
        st.session_state["last_level"] = None

    elif level == "distrito":
        st.session_state["selected_distrito_id"] = None
        st.session_state["selected_iso_ids"] = set()
        st.session_state["selected_censo_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["iso_next_mode"] = "quadra"
        st.session_state["quadra_filter_col"] = CENSO_ID
        st.session_state["quadra_filter_uids"] = set()

    elif level == "isocrona":
        st.session_state["selected_iso_ids"] = set()
        st.session_state["selected_censo_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["iso_next_mode"] = "quadra"
        st.session_state["quadra_filter_col"] = CENSO_ID
        st.session_state["quadra_filter_uids"] = set()

    elif level == "censo":
        st.session_state["selected_censo_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["quadra_filter_col"] = CENSO_ID
        st.session_state["quadra_filter_uids"] = set()

    elif level == "quadra":
        st.session_state["selected_quadra_ids"] = set()

    elif level == "final":
        pass


def _prev_level(level: str) -> Optional[str]:
    if level not in LEVELS:
        return None
    i = LEVELS.index(level)
    if i <= 0:
        return None
    return LEVELS[i - 1]


def _back_one_level() -> None:
    cur = st.session_state.get("level", "subpref")
    prev = _prev_level(cur)
    if prev:
        reset_to(prev)


def _toggle_in_set(key: str, value: Any) -> None:
    s: Set[Any] = st.session_state.get(key, set()) or set()
    if value in s:
        s.remove(value)
    else:
        s.add(value)
    st.session_state[key] = s


def debug_layer_schema(layer_key: str, g: Optional["gpd.GeoDataFrame"]) -> None:
    if not st.session_state.get("debug_schema", False):
        return
    if g is None:
        st.write(f"[debug] layer={layer_key}: None")
        return
    st.write(f"[debug] layer={layer_key} shape={g.shape}")
    st.write(f"[debug] layer={layer_key} cols={list(g.columns)}")
    try:
        st.write(f"[debug] layer={layer_key} crs={getattr(g, 'crs', None)}")
    except Exception:
        pass


# =============================================================================
# DRIVE / LOCAL IO
# =============================================================================
SECRETS_KEYS = {
    "subpref": "PB_SUBPREF_FILE_ID",
    "dist": "PB_DISTRITO_FILE_ID",
    "iso": "PB_ISOCRONAS_FILE_ID",
    "quadra": "PB_QUADRAS_FILE_ID",
    "lote": "PB_LOTES_FILE_ID",
    "censo": "PB_CENSO_FILE_ID",
}

FALLBACK_URLS = {
    "subpref": "https://drive.google.com/file/d/1vPY34cQLCoGfADpyOJjL9pNCYkVrmSZA/view?usp=drive_link",
    "dist": "https://drive.google.com/file/d/1K-t2BiSHN_D8De0oCFxzGdrEMhnGnh10/view?usp=drive_link",
    "iso": "https://drive.google.com/file/d/18ukyzMiYQ6vMqrU6-ctaPFbXMPX9XS9i/view?usp=drive_link",
    "quadra": "https://drive.google.com/file/d/17VaA-MlITota7shvbN8mNgf2MjWvUbVW/view?usp=drive_link",
    "lote": "https://drive.google.com/file/d/1oTFAZff1mVAWD6KQTJSz45I6B6pi6ceP/view?usp=drive_link",
    "censo": "https://drive.google.com/file/d/1APp7fxT2mgTpegVisVyQwjTRWOPz6Rgn/view?usp=drive_link",
}

_DRIVE_ID_RE_1 = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")
_DRIVE_ID_RE_2 = re.compile(r"[?&]id=([a-zA-Z0-9_-]+)")


def _get_secret(key: str) -> str:
    try:
        return str(st.secrets.get(key, "")).strip()
    except Exception:
        return ""


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
    url = "https://drive.google.com/uc?export=download"

    def get_confirm_token(resp) -> Optional[str]:
        for k, v in resp.cookies.items():
            if k.startswith("download_warning"):
                return v
        return None

    response = session.get(url, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        response = session.get(url, params={"id": file_id, "confirm": token}, stream=True)

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


def get_drive_raw(layer_key: str) -> str:
    ui_key = f"drive_{layer_key}_raw"
    raw_ui = str(st.session_state.get(ui_key, "")).strip()
    if raw_ui:
        return raw_ui

    secret_key = SECRETS_KEYS.get(layer_key, "")
    raw_secret = _get_secret(secret_key) if secret_key else ""
    if raw_secret:
        return raw_secret

    return str(FALLBACK_URLS.get(layer_key, "")).strip()


def local_layer_path(layer_key: str) -> Path:
    return DATA_CACHE_DIR / LOCAL_FILENAMES[layer_key]


def layer_available_locally(layer_key: str) -> bool:
    p = local_layer_path(layer_key)
    return p.exists() and p.stat().st_size > 0


def ensure_local_layer(layer_key: str) -> Path:
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
# READ / FILTER
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def read_gdf_parquet(path: str) -> Optional["gpd.GeoDataFrame"]:
    if gpd is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    gdf = gpd.read_parquet(p)
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


def read_layer(layer_key: str) -> Optional["gpd.GeoDataFrame"]:
    try:
        p = ensure_local_layer(layer_key)
    except Exception as e:
        st.error(str(e))
        return None

    try:
        meta = (str(p), float(p.stat().st_mtime), int(p.stat().st_size))
    except Exception:
        meta = (str(p), 0.0, 0)

    cache: Dict[str, Any] = st.session_state.get("_layer_cache", {})
    cache_meta: Dict[str, Any] = st.session_state.get("_layer_cache_meta", {})

    if layer_key in cache and cache_meta.get(layer_key) == meta:
        g_cached = cache.get(layer_key)
        if g_cached is not None:
            return g_cached

    g = read_gdf_parquet(str(p))
    if g is None or g.empty:
        st.error(f"Layer '{layer_key}' vazia/erro ao ler ({p.name}).")
        return None

    g = standardize_columns(g)
    g = _drop_bad_geoms(g)
    g = normalize_id_cols(g, LAYER_ID_COLS.get(layer_key, []))

    if layer_key == "quadra":
        if QUADRA_ID in g.columns:
            g[QUADRA_ID] = g[QUADRA_ID].map(lambda x: normalize_quadra_id(x, 6))
        if ISO_ID in g.columns and QUADRA_ID in g.columns:
            g[QUADRA_UID] = [make_quadra_uid(i, q) for i, q in zip(g[ISO_ID], g[QUADRA_ID])]

    cache[layer_key] = g
    cache_meta[layer_key] = meta
    st.session_state["_layer_cache"] = cache
    st.session_state["_layer_cache_meta"] = cache_meta

    # ✅ debug: mostra meta pra validar se é o arquivo certo
    if st.session_state.get("debug_schema", False):
        st.write(f"[debug] loaded {layer_key}: path={p} mtime={p.stat().st_mtime} size={p.stat().st_size}")
        debug_layer_schema(layer_key, g)

    return g


def subset_by_parent_multi(child: "gpd.GeoDataFrame", parent_col: str, parent_vals: Set[Any]) -> "gpd.GeoDataFrame":
    if child is None or child.empty:
        return child
    if parent_col not in child.columns or not parent_vals:
        return child.iloc[0:0].copy()
    pset = {v for v in (_id_to_str(x) for x in parent_vals) if v is not None}
    if not pset:
        return child.iloc[0:0].copy()
    return child[child[parent_col].isin(list(pset))]


def subset_by_id_multi(gdf: "gpd.GeoDataFrame", id_col: str, ids: Set[Any]) -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty:
        return gdf
    if id_col not in gdf.columns or not ids:
        return gdf.iloc[0:0].copy()
    iset = {v for v in (_id_to_str(x) for x in ids) if v is not None}
    if not iset:
        return gdf.iloc[0:0].copy()
    return gdf[gdf[id_col].isin(list(iset))]


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
# UI helpers / flow
# =============================================================================
def _variables_for_level(level: str) -> List[str]:
    if level == "subpref":
        return ["Subprefeituras"]
    if level == "distrito":
        return ["Distritos"]
    if level == "isocrona":
        return ["Isócronas", "Isócronas (classes)"]
    if level == "censo":
        return ["Setor censitário"]
    if level == "quadra":
        return ["Quadras", "Cluster"]
    if level == "final":
        return ["Lotes"]
    return ["Nível"]


def ensure_variable_for_level(level: str) -> None:
    opts = _variables_for_level(level)
    cur = st.session_state.get("variable")
    if cur not in opts:
        st.session_state["variable"] = opts[0]


def variable_panel() -> None:
    lvl = st.session_state.get("level", "subpref")
    ensure_variable_for_level(lvl)
    st.selectbox("Variável", options=_variables_for_level(lvl), key="variable", on_change=mark_ui_action)


def control_panel() -> None:
    lvl = st.session_state.get("level", "subpref")
    prev = None
    if lvl in LEVELS:
        i = LEVELS.index(lvl)
        prev = LEVELS[i - 1] if i > 0 else None

    c1, c2 = st.columns(2)
    with c1:
        if prev is None:
            st.button("Subprefeituras", disabled=True, use_container_width=True)
        else:
            st.button(
                prev.capitalize(),
                type="primary",
                use_container_width=True,
                on_click=lambda: (mark_ui_action(), reset_to(prev)),
            )
    with c2:
        st.button(
            "Reset",
            type="primary",
            use_container_width=True,
            on_click=lambda: (mark_ui_action(), reset_to("subpref")),
        )

    st.divider()
    st.subheader("Variável", anchor=False)
    variable_panel()

    st.divider()
    st.subheader("Debug", anchor=False)
    st.checkbox("Mostrar schema/colunas carregadas (debug)", key="debug_schema")

    st.divider()
    st.subheader("Ações", anchor=False)

    if lvl == "isocrona":
        ok = len(st.session_state.get("selected_iso_ids", set()) or set()) > 0
        st.radio(
            "Próximo nível",
            options=["Quadras", "Setor censitário"],
            index=0 if st.session_state.get("iso_next_mode") == "quadra" else 1,
            key="_iso_next_ui",
            on_change=mark_ui_action,
        )
        st.session_state["iso_next_mode"] = "quadra" if st.session_state["_iso_next_ui"] == "Quadras" else "censo"
        st.button(
            "Prosseguir",
            use_container_width=True,
            disabled=not ok,
            on_click=lambda: (mark_ui_action(), _go_from_isos_next()),
        )

    if lvl == "censo":
        okc = len(st.session_state.get("selected_censo_ids", set()) or set()) > 0
        st.button(
            "Prosseguir → Quadras (censo_id)",
            use_container_width=True,
            disabled=not okc,
            on_click=lambda: (mark_ui_action(), _go_from_censo_to_quadras()),
        )


def _go_from_isos_next() -> None:
    mode = st.session_state.get("iso_next_mode", "quadra")
    if mode == "censo":
        st.session_state["level"] = "censo"
        st.session_state["selected_censo_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["quadra_filter_col"] = CENSO_ID
        st.session_state["quadra_filter_uids"] = set()
        _final_reset()
    else:
        st.session_state["level"] = "quadra"
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["quadra_filter_col"] = CENSO_ID
        st.session_state["quadra_filter_uids"] = set()
        _final_reset()


def _go_from_censo_to_quadras() -> None:
    sel_censo: Set[str] = st.session_state.get("selected_censo_ids", set()) or set()
    if not sel_censo:
        st.warning("Selecione ao menos um setor (censo_id).")
        return
    # ✅ força filtro por censo_id
    st.session_state["quadra_filter_col"] = CENSO_ID
    st.session_state["quadra_filter_uids"] = sel_censo
    st.session_state["selected_quadra_ids"] = set()
    st.session_state["level"] = "quadra"
    _final_reset()


# =============================================================================
# MAP RENDER (somente parte essencial do problema: nível quadra)
# =============================================================================
def render_map_panel() -> None:
    level = st.session_state.get("level", "subpref")
    ensure_variable_for_level(level)

    # ... (demais níveis iguais ao seu arquivo; omiti aqui para focar)
    # IMPORTANTE: manter seu código completo. Abaixo está o bloco "quadra" corrigido.

    if level != "quadra":
        st.info("Para manter esta resposta objetiva, o bloco completo de outros níveis não foi reimpresso aqui.")
        st.stop()

    iso_ids = {v for v in (_id_to_str(x) for x in st.session_state.get("selected_iso_ids", set())) if v}
    if not iso_ids:
        st.warning("Selecione ao menos uma isócrona.")
        st.stop()

    mode = st.session_state.get("iso_next_mode", "quadra")
    filter_ids: Set[str] = st.session_state.get("quadra_filter_uids", set()) or set()
    filter_col: str = st.session_state.get("quadra_filter_col", CENSO_ID)

    g_quad = read_layer("quadra")
    if g_quad is None:
        st.stop()

    debug_layer_schema("quadra", g_quad)

    # ✅ define o id do mapa
    if mode == "censo":
        id_col_map = QUADRA_ID
    else:
        id_col_map = QUADRA_UID if QUADRA_UID in g_quad.columns else QUADRA_ID
    st.session_state["_quadra_id_col_map"] = id_col_map

    # ✅ g_show por duas vias
    if mode == "censo":
        censo_col = QUADRA_PARENTS["censo"]  # "censo_id"
        if censo_col not in g_quad.columns:
            st.error(
                f"Modo 'censo' ativo, mas a coluna '{censo_col}' não foi encontrada em Quadras.parquet.\n"
                f"Colunas carregadas: {list(g_quad.columns)}\n"
                "Isso costuma indicar cache/arquivo diferente do esperado ou nome de coluna com espaços/alias não tratado."
            )
            st.stop()

        if not filter_ids or filter_col != CENSO_ID:
            st.warning("Nenhum filtro por censo_id encontrado. Volte ao Setor censitário e selecione setores.")
            g_show = g_quad.iloc[0:0].copy()
        else:
            g_show = subset_by_id_multi(g_quad, censo_col, filter_ids)

    else:
        iso_col = QUADRA_PARENTS["iso"]  # "iso_id"
        if iso_col not in g_quad.columns:
            st.error(f"Quadras.parquet não tem '{iso_col}'. Colunas: {list(g_quad.columns)}")
            st.stop()
        g_show = subset_by_parent_multi(g_quad, iso_col, iso_ids)

    st.write("Quadras filtradas:", 0 if g_show is None else len(g_show))
    st.stop()


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

    # ------------------------------------------------------------
    # Pré-consumo do clique do mapa:
    # - Consome clique do RUN anterior antes do render do mapa.
    # - Só consome se o mapa anterior foi renderizado no mesmo nível atual.
    # ------------------------------------------------------------
    ui_sig = int(st.session_state.get("_ui_action_sig", 0))
    ui_seen = int(st.session_state.get("_ui_action_sig_seen", 0))
    ui_action = ui_sig != ui_seen
    st.session_state["_ui_action_sig_seen"] = ui_sig

    if ui_action:
        st.session_state["last_click_sig"] = ""

    cur_level = st.session_state.get("level", "subpref")
    rendered_level = st.session_state.get("_map_level_rendered")
    map_state_prev = st.session_state.get(MAP_KEY, {}) or {}

    allow_click = (not ui_action) and (rendered_level == cur_level)

    if allow_click and isinstance(map_state_prev, dict) and map_state_prev:
        consume_map_event(cur_level, map_state_prev, allow_click=True)

    sanitize_level_state()

    left, right = st.columns([4, 1], gap="large")
    with left:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        render_map_panel()
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        control_panel()
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
```__

